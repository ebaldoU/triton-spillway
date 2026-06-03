import os
from pathlib import Path

BASE_URI = os.environ.get(
    "TRITON_BASE_URI",
    str(Path(__file__).parent.parent / "triton_results"),
)

# Directorio raíz donde están las carpetas de GeoTIFFs fuente (datos1/, datos2/, ...)
GTIFF_BASE_URI = os.environ.get(
    "TRITON_GTIFF_DIR",
    str(Path(BASE_URI).parent.parent),
)

# Directorio de salida para exportaciones
OUTPUT_DIR = Path(os.environ.get("TRITON_OUTPUT_DIR", str(Path(BASE_URI).parent / "export")))

DATASET_ALIASES = {
    "datos1": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19800728_19800801",
    "datos2": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19810515_19810519",
    "datos3": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19820706_19820710",
    "datos4": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19830309_19830313",
    "datos5": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19840619_19840623",
    "datos6": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19851011_19851015",
    "datos7": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19860905_19860909",
    "datos8":  "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19870918_19870922",
    "datos9":  "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19880420_19880424",
    "datos10": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19890509_19890513",
    "datos11": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19900427_19900501",
    "datos12": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19910526_19910530",
    "datos13": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19920314_19920318",
    "datos14": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19931111_19931115",
    "datos15": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19940421_19940425",
    "datos16": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19950530_19950603",
    "datos17": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19960815_19960819",
    "datos18": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19970620_19970624",
    "datos19": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19980709_19980713",
    "datos20": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_19990810_19990814",
    "datos21": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20000915_20000919",
    "datos22": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20010507_20010511",
    "datos23": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20021017_20021021",
    "datos24": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20030429_20030503",
    "datos25": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20040708_20040712",
    "datos26": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20051006_20051010",
    "datos27": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20060516_20060520",
    "datos28": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20070919_20070923",
    "datos29": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20080602_20080606",
    "datos30": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20090807_20090811",
    "datos31": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20100510_20100514",
    "datos32": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20110526_20110530",
    "datos33": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20120504_20120508",
    "datos34": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20130509_20130513",
    "datos35": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20140518_20140522",
    "datos36": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20150501_20150505",
    "datos37": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20160606_20160610",
    "datos38": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20170905_20170909",
    "datos39": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20180522_20180526",
    "datos40": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_1980_2019_20190430_20190504",
    "datos41": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20200501_20200505",
    "datos42": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20210510_20210514",
    "datos43": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20220915_20220919",
    "datos44": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20230425_20230429",
    "datos45": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20240620_20240624",
    "datos46": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20250519_20250523",
    "datos47": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20260414_20260418",
    "datos48": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20270511_20270515",
    "datos49": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20281114_20281118",
    "datos50": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20290531_20290604",
    "datos51": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20300423_20300427",
    "datos52": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20310412_20310416",
    "datos53": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20320521_20320525",
    "datos54": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20330506_20330510",
    "datos55": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20341015_20341019",
    "datos56": "output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet_2020_2059_20350408_20350412",
}
DATASET_LABELS = {v: k for k, v in DATASET_ALIASES.items()}


def resolve_dataset(name: str) -> str:
    """datos1 → nombre real del directorio TileDB."""
    return DATASET_ALIASES.get(name, name)


def dataset_label(name: str) -> str:
    """Nombre real del directorio → alias corto para mostrar."""
    return DATASET_LABELS.get(name, name)
