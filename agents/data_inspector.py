import pandas as pd


class DataInspector:

    def inspect(self, df: pd.DataFrame):

        report = {}

        report["rows"] = df.shape[0]
        report["columns"] = df.shape[1]

        report["missing_values"] = int(df.isnull().sum().sum())

        report["duplicate_rows"] = int(df.duplicated().sum())

        report["numeric_columns"] = list(
            df.select_dtypes(include="number").columns
        )

        report["categorical_columns"] = list(
            df.select_dtypes(exclude="number").columns
        )

        report["data_types"] = {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        }

        return report