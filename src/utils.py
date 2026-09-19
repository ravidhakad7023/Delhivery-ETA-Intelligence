from pathlib import Path


class Utils:

    @staticmethod
    def save_dataframe(df, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if path.suffix == ".csv":
            df.to_csv(path, index=False)

        elif path.suffix == ".parquet":
            df.to_parquet(path, index=False)

        print(f"Saved: {path}")