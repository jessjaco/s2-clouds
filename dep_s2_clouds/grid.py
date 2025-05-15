from pathlib import Path
import geopandas as gpd

s2_grid_path = Path("data/s2_grid.gpkg")
if not s2_grid_path.exists():
    GADM = gpd.read_file(
        "https://dep-public-staging.s3.us-west-2.amazonaws.com/aoi/aoi.gpkg",
        layer="aoi",
    )

    s2_grid = gpd.read_file(
        "https://hls.gsfc.nasa.gov/wp-content/uploads/2016/03/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml"
    )
    s2_grid = s2_grid.loc[s2_grid.sjoin(GADM, how="inner").index]
    s2_grid.to_file(s2_grid_path)

s2_grid = gpd.read_file(s2_grid_path).set_index("Name")
