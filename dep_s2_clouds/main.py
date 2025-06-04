from concurrent.futures import process
from datetime import datetime
from itertools import product
import json
from logging import getLogger, Logger
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Annotated
import warnings

import boto3
from cloud_logger import CsvLogger, filter_by_log, S3Handler
from dep_tools.aws import object_exists, s3_dump
from dep_tools.loaders import OdcLoader, StacLoader
from dep_tools.namers import S3ItemPath
from dep_tools.processors import Processor
from dep_tools.stac_utils import StacCreator
from dep_tools.task import Task
from dep_tools.utils import search_across_180
from dep_tools.writers import AwsStacWriter, Writer, AwsDsCogWriter
from omnicloudmask import predict_from_array
from odc.stac import configure_s3_access
from pystac import Item
import pystac_client
import typer
import xarray as xr

from dep_s2_clouds.grid import s2_grid


BUCKET = "dep-public-staging"
DATASET_ID = "ocm"
VERSION = "0.1.0"
BATCH_SIZE = int(os.environ.get("DEP_BATCH_SIZE", 5))


app = typer.Typer()


def parse_datetime(datetime):
    years = datetime.split("_")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{datetime} is not a valid value for --datetime")
    return years


def bool_parser(raw: str):
    return False if raw == "False" else True


class OCMProcessor(Processor):
    def __init__(self, batch_size=BATCH_SIZE, inference_dtype="bf16", **kwargs):
        self._batch_size = batch_size
        self._inference_dtype = inference_dtype
        self._kwargs = kwargs

    def process(self, ds):
        mask_xr = xr.zeros_like(ds.red.astype("uint8").drop_attrs()).rio.write_crs(
            ds.red.rio.crs
        )
        mask_xr.values = predict_from_array(
            ds.squeeze().to_array().values,
            batch_size=self._batch_size,
            inference_dtype=self._inference_dtype,
            # gdrive links have daily limits on download
            model_download_source="hugging_face",
            **self._kwargs,
        )
        return mask_xr.to_dataset(name="mask")


class S2DailyItemPath(S3ItemPath):
    def __init__(self, time: datetime | None = None, **kwargs):
        super().__init__(time=time, **kwargs)

    def _folder(self, item_id) -> str:
        return f"{self._folder_prefix}/{self._format_item_id(item_id)}/{self.time:%Y/%m/%d}"

    def basename(self, item_id) -> str:
        return f"{self.item_prefix}_{self._format_item_id(item_id, join_str='_')}_{self.time:%Y-%m-%d}"


@app.command()
def process_s2_mask(s2_id: Annotated[str, typer.Option()]):
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    item = Item.from_file(
        f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{s2_id}"
    )
    # no leading zeroes on 0-9 tile ids in stac item
    tile_id = re.sub(r"-(\w{4})$", r"0\1", item.properties["grid:code"][-5:])
    itempath = S2DailyItemPath(
        bucket=BUCKET,
        sensor="s2",
        dataset_id=DATASET_ID,
        version=VERSION,
        time=item.get_datetime(),
    )
    if not object_exists(bucket=BUCKET, key=itempath.stac_path(tile_id)):
        try:
            loader = OdcLoader(
                dtype="uint16",
                bands=["red", "green", "nir08"],
                anchor="center",
            )
            return ItemStacTask(
                id=tile_id,
                loader=loader,
                processor=OCMProcessor(),
                writer=AwsDsCogWriter(itempath),
                stac_creator=StacCreator(itempath),
                stac_writer=AwsStacWriter(itempath),
            ).run(item)

        except Exception as e:
            warnings.warn("Error from one of the dailies, check the output logs")
            daily_log_path = Path(itempath.log_path()).with_suffix(".error.txt")
            boto3_client = boto3.client("s3")

            s3_dump(
                data=traceback.format_exc(),
                bucket=BUCKET,
                key=str(daily_log_path),
                client=boto3_client,
            )
    else:
        return []


@app.command()
def process_ids(
    s2_cell: Annotated[str, typer.Option()], datetime: Annotated[str, typer.Option()]
):
    itempath = S3ItemPath(
        bucket=BUCKET,
        sensor="s2",
        dataset_id=DATASET_ID,
        version=VERSION,
        time=str(datetime).replace("/", "_"),
    )
    logger = CsvLogger(
        name="ocm",
        path=f"{itempath.bucket}/{itempath.log_path()}",
        overwrite=False,
        header="time|index|status|paths|comment\n",
        cloud_handler=S3Handler,
    )
    try:
        paths = [process_s2_mask(s2_id) for s2_id in ids(s2_cell, datetime)]
    except Exception as e:
        logger.error([s2_cell, "error", [], f'"{e}"'])
        raise e

    logger.info([s2_cell, "complete", paths])


@app.command()
def ids(
    s2_cell: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    output_json: Annotated[str, typer.Option(parser=bool_parser)] = False,
) -> list | None:
    client = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    items = search_across_180(
        region=s2_grid.loc[[s2_cell]],
        client=client,
        collections=["sentinel-2-l2a"],
        query={"grid:code": {"eq": f"MGRS-{s2_cell.lstrip('0')}"}},
        datetime=datetime,
    )
    ids = [item.id for item in items]
    return json.dump(ids, sys.stdout) if output_json else ids


@app.command()
def cells_and_years(
    datetime: Annotated[str, typer.Option(parser=parse_datetime)],
    select_tiles_only: Annotated[str, typer.Option(parser=bool_parser)] = "False",
):
    select_tiles = [
        "59NLG",
        "56MKU",
        "01KGU",
        "07LFL",
        "01KHV",
        "56MQU",
        "60KWF",
        "59NLG",
        "54LYR",
        "01KFS",
        "55MEM",
        "57LXK",
        "57NVH",
        "58KEB",
        "58KHF",
        "59NQB",
        "60KXF",
        "60LYR",
    ]
    grid = s2_grid.loc[select_tiles] if select_tiles_only else s2_grid
    output = []
    for year in datetime:
        itempath = S3ItemPath(
            bucket=BUCKET,
            sensor="s2",
            dataset_id=DATASET_ID,
            version=VERSION,
            time=str(year).replace("/", "_"),
        )
        logger = CsvLogger(
            name="ocm",
            path=f"{itempath.bucket}/{itempath.log_path()}",
            overwrite=False,
            header="time|index|status|paths|comment\n",
            cloud_handler=S3Handler,
        )

        grid_subset = filter_by_log(
            grid, logger.parse_log(), retry_errors=False, parse_index=False
        )
        output += [dict(s2_cell=cell, datetime=year) for cell in grid_subset.index]
    json.dump(output, sys.stdout)


def copy_stac_properties(item, ds):
    if "stac_properties" in ds.attrs:
        ds.attrs["stac_properties"] = {
            **ds.attrs.get("stac_properties"),
            **item.properties,
        }
    else:
        ds.attrs["stac_properties"] = item.properties

    ds.attrs["stac_properties"]["start_datetime"] = ds.attrs["stac_properties"][
        "datetime"
    ]
    ds.attrs["stac_properties"]["end_datetime"] = ds.attrs["stac_properties"][
        "datetime"
    ]
    return ds


class ItemStacTask(Task):
    def __init__(
        self,
        id: str,
        loader: StacLoader,
        processor: Processor,
        writer: Writer,
        post_processor: Processor | None = None,
        stac_creator: StacCreator | None = None,
        stac_writer: Writer | None = None,
        logger: Logger = getLogger(),
    ):
        super().__init__(
            task_id=id, loader=loader, processor=processor, writer=writer, logger=logger
        )
        self.post_processor = post_processor
        self.stac_creator = stac_creator
        self.stac_writer = stac_writer

    def run(self, item):
        input_data = self.loader.load([item], areas=None)

        output_data = copy_stac_properties(item, self.processor.process(input_data))

        if self.post_processor is not None:
            output_data = self.post_processor.process(output_data)

        paths = self.writer.write(output_data, self.id)

        if self.stac_creator is not None and self.stac_writer is not None:
            stac_item = self.stac_creator.process(output_data, self.id)
            self.stac_writer.write(stac_item, self.id)

        return paths


if __name__ == "__main__":
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    # process_s2_mask("S2B_59WNV_20250513_0_L2A")
    app()
