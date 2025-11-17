import logging
import argparse
import numpy as np
import pandas as pd


from typing import List
from pathlib import Path


logger = logging.getLogger(__name__)


def read_data(path: Path | str) -> pd.DataFrame:
    """
    Read the raw CSV file and fix lines where County and State were merged
    due to an extra comma in the County field.

    Parameters
    ----------
    path : Path | str
        Path to the input CSV.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    logger.info("Reading header from %s", path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    expected_cols = len(header)

    try:
        county_idx = header.index("County")
        state_idx = header.index("State")
    except ValueError as exc:
        raise RuntimeError(
            ("Expected columns 'County' and 'State' "
             "are missing from the file header.")
        ) from exc

    def fix_bad_line(line: List[str]) -> List[str]:
        """
        Callable for pandas on_bad_lines.
        If a line has one extra column (County split by comma),
        merge County and State back together.
        """
        n = len(line)

        if n == expected_cols + 1:
            merged_county = f"{line[county_idx]},{line[state_idx]}"
            fixed_line = (
                line[:county_idx]
                + [merged_county]
                + line[state_idx + 1:]
            )
            return fixed_line

        # For all other cases, keep the line as-is.
        return line

    logger.info("Reading full CSV with bad line fixer")
    df = pd.read_csv(
        path,
        engine="python",
        encoding="latin-1",
        on_bad_lines=fix_bad_line,
    )
    return df


def drop_unrecoverable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that cannot be reliably recovered:
      - Rows with invalid or missing row_id (from 'Unnamed: 0')
      - Rows where ID suffix cannot be parsed as numeric.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Clean up row_id if present
    if "Unnamed: 0" in df.columns:
        logger.info("Cleaning 'Unnamed: 0' -> 'row_id'")
        df = df.rename(columns={"Unnamed: 0": "row_id"})
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
        before = len(df)
        df = df[df["row_id"].notna()]
        logger.info("Dropped %d rows with invalid row_id", before - len(df))

    # Filter rows with non-parsable IDs (keep suffix only)
    if "ID" in df.columns:
        logger.info("Filtering rows with invalid IDs")
        id_suffix = df["ID"].astype(str).str.split("-").str[-1]
        id_numeric = pd.to_numeric(id_suffix, errors="coerce")

        before = len(df)
        mask = id_numeric.notna()
        df = df.loc[mask].copy()
        df["ID"] = id_suffix[mask]
        logger.info("Dropped %d rows with invalid ID", before - len(df))
    else:
        logger.warning("'ID' column not found; skipping ID-based filtering")

    return df


def fix_shifted_county_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix records where 'County' accidentally contains the value 'NC'
    (which should be the State). For these rows, shift all columns to the
    right starting from the column after 'County' and set County to NaN.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if "County" not in df.columns:
        logger.warning("'County' column not found; skipping county shift fix")
        return df

    df = df.copy()
    county_col_idx = df.columns.get_loc("County")
    cols_after = df.columns[county_col_idx + 1:]

    shift_idx = df.index[df["County"] == "NC"]
    logger.info("Found %d records with County == 'NC' to fix", len(shift_idx))

    if len(shift_idx) == 0:
        return df

    df.loc[shift_idx, cols_after] = (
        df.loc[shift_idx, cols_after].shift(1, axis=1)
    )
    df.loc[shift_idx, "County"] = np.nan
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column data types:
      - datetime columns -> datetime64[ns]
      - numeric columns  -> float (coerce errors to NaN)
      - boolean-like columns -> 0/1 integers
      - categorical columns -> string

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    datetime_cols = [
        "Start_Time",
        "End_Time",
        "Weather_Timestamp",
    ]

    numeric_cols = [
        "Severity",
        "Start_Lat",
        "Start_Lng",
        "End_Lat",
        "End_Lng",
        "Distance(mi)",
        "Wind_Chill(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
    ]

    categorical_cols = [
        "ID",
        "Source",
        "Description",
        "Street",
        "City",
        "County",
        "State",
        "Zipcode",
        "Country",
        "Timezone",
        "Airport_Code",
        "Wind_Direction",
        "Weather_Condition",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
        "Temperature_Range(F)"
    ]

    bool_cols = [
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop",
    ]

    for col in datetime_cols:
        if col in df.columns:
            logger.info("Converting %s to datetime", col)
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    for col in numeric_cols:
        if col in df.columns:
            logger.info("Converting %s to numeric", col)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _parse_bool_series(s: pd.Series) -> pd.Series:
        s_str = s.astype(str).str.strip().str.lower()
        true_vals = {"true", "1", "t", "y"}
        false_vals = {"false", "0", "f", "n", ""}

        result = pd.Series(pd.NA, index=s.index, dtype="boolean")
        result = result.mask(s_str.isin(true_vals), True)
        result = result.mask(s_str.isin(false_vals), False)
        result = result.fillna(False)
        return result.astype("boolean").astype("Int8")

    for col in bool_cols:
        if col in df.columns:
            logger.info("Converting %s to boolean 0/1", col)
            df[col] = _parse_bool_series(df[col])

    for col in categorical_cols:
        if col in df.columns:
            logger.info("Casting %s to string", col)
            df[col] = df[col].astype("string")

    return df


def split_range_columns(df):
    """
    Split columns with range values (e.g., '10-20') into two separate columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df['min_temperature_f'] = (
        df['Temperature_Range(F)']
        .str.split('-').str[0]
        .astype(float, errors='ignore')
    )
    df['max_temperature_f'] = (
        df['Temperature_Range(F)']
        .str.split('-').str[1]
        .astype(float, errors='ignore')
    )
    df = df.drop('Temperature_Range(F)', axis=1)

    return df


def unify_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names:
      - strip whitespace
      - lowercase
      - replace spaces with underscores
      - remove non-alphanumeric characters (excluding underscore)
    """
    df = df.copy()
    logger.info("Normalizing column names")
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    return df


def unify_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize categorical variable values:
      - cast to string
      - strip whitespace
      - lowercase
      - replace spaces with underscores
    """
    df = df.copy()
    cat_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    logger.info(
        "Normalizing categorical variables for %d columns",
        len(cat_cols)
    )

    for col in cat_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and preprocess US accidents dataset."
    )
    default_base = (
        Path(__file__).resolve().parent
        if "__file__" in globals()
        else Path.cwd()
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            default_base / ".." / "data" / "raw" / "US_Accidents_March23.csv"
        ),
        help=(
            "Path to raw input CSV "
            "(default: ../data/raw/US_Accidents_March23.csv)"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_base / ".." / "data" / "processed" / "accidents.csv",
        help=(
            "Path to save processed CSV "
            "(default: ../data/processed/accidents.csv)"
        )
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    logger.info("Input path:  %s", input_path)
    logger.info("Output path: %s", output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = read_data(input_path)
    logger.info("Raw data shape: %s", df_raw.shape)

    df = drop_unrecoverable_rows(df_raw)
    logger.info("After dropping unrecoverable rows: %s", df.shape)

    df = fix_shifted_county_records(df)
    logger.info("After fixing shifted County records: %s", df.shape)

    df = convert_types(df)
    df = split_range_columns(df)
    df = unify_column_names(df)
    df = unify_categorical_variables(df)

    logger.info("Final dataframe shape: %s", df.shape)
    logger.info("Saving processed data to %s", output_path)
    df.to_csv(output_path, index=False)
    logger.info("Done")


if __name__ == "__main__":
    main()
