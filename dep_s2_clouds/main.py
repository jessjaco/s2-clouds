import json
from logging import getLogger, Logger
from pathlib import Path
import sys
import traceback
from typing import Annotated
import warnings

import boto3
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

app = typer.Typer()


class OCMProcessor(Processor):
    def __init__(self, batch_size=5, inference_dtype="bf16", **kwargs):
        self._batch_size = batch_size
        self._inference_dtype = inference_dtype
        self._kwargs = kwargs

    def process(self, ds):
        mask = predict_from_array(
            ds.squeeze().to_array().values,
            batch_size=self._batch_size,
            inference_dtype=self._inference_dtype,
            **self._kwargs,
        )
        mask_xr = xr.zeros_like(ds.red.astype("uint8"))
        mask_xr.values = mask
        return mask_xr


@app.command()
def process_s2_mask(s2_id: Annotated[str, typer.Option()]):
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    item = Item.from_file(
        f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items/{s2_id}"
    )
    itempath = S3ItemPath(
        bucket=BUCKET,
        sensor="s2",
        dataset_id=DATASET_ID,
        version=VERSION,
        time=item.get_datetime(),
    )
    if not object_exists(bucket=BUCKET, key=itempath.stac_path(s2_id)):
        try:
            loader = OdcLoader(
                dtype="uint16",
                bands=["red", "green", "nir08"],
                anchor="center",
            )
            return ItemStacTask(
                id=s2_id,
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


@app.command()
def print_ids(
    s2_cell: Annotated[str, typer.Option()], datetime: Annotated[str, typer.Option()]
):
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    client = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    items = search_across_180(
        region=s2_grid.loc[[s2_cell]],
        client=client,
        collections=["sentinel-2-c1-l2a"],
        query={"grid:code": f"MGRS-{s2_cell}"},
        datetime=datetime,
    )
    item_ids = [{"s2_id": item.id} for item in items]
    json.dump(item_ids, sys.stdout)


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

        breakpoint()
        if self.post_processor is not None:
            output_data = self.post_processor.process(output_data)

        paths = self.writer.write(output_data, self.id)

        if self.stac_creator is not None and self.stac_writer is not None:
            stac_item = self.stac_creator.process(output_data, self.id)
            self.stac_writer.write(stac_item, self.id)

        return paths


if __name__ == "__main__":
    process_s2_mask("S2A_T60KXF_20210503T221937_L2A")
    # app()
