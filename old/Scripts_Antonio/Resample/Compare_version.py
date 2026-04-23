import pandas as pd
df_antonio_in_final = pd.read_parquet('c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/backup/Antonio/parquets/diatrend/Glucose_measurements_antonio.parquet')
df_ciro = pd.read_parquet('c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/backup/Antonio/parquets/diatrend/v4/Glucose_measurements_v4_by_Ciro.parquet')

df_ciro["Patient_ID"] = df_ciro["Patient_ID"].astype("string")
df_ciro["5min"] = pd.to_datetime(df_ciro["5min"], errors="coerce")
df_ciro["15min"] = pd.to_datetime(df_ciro["15min"], errors="coerce")
df_ciro["Measurement"] = pd.to_numeric(df_ciro["Measurement"], errors="coerce")
df_ciro["Measurement_date"] = df_ciro["Measurement_date"].astype("string")
df_ciro["Measurement_time"] = df_ciro["Measurement_time"].astype("string")

df_antonio_in_final["Patient_ID"] = df_antonio_in_final["Patient_ID"].astype("string")
df_antonio_in_final["5min"] = pd.to_datetime(df_antonio_in_final["5min"], errors="coerce")
df_antonio_in_final["15min"] = pd.to_datetime(df_antonio_in_final["15min"], errors="coerce")
df_antonio_in_final["Measurement"] = pd.to_numeric(df_antonio_in_final["Measurement"], errors="coerce")
df_antonio_in_final["Measurement_date"] = df_antonio_in_final["Measurement_date"].astype("string")
df_antonio_in_final["Measurement_time"] = df_antonio_in_final["Measurement_time"].astype("string")

pdid_list = df_antonio_in_final['Patient_ID'].unique()
df_ciro_filter = df_ciro[df_ciro['Patient_ID'].isin(pdid_list)]
df_ciro_filter.sort_values(by=['Patient_ID', 'Measurement_date', 'Measurement_time', '5min', '15min', 'Measurement'], inplace=True, ignore_index=True)
df_ciro_filter.reset_index(drop=True, inplace=True)

df_antonio_in_final.sort_values(by=['Patient_ID', 'Measurement_date', 'Measurement_time', '5min', '15min', 'Measurement'], inplace=True, ignore_index=True)
df_antonio_in_final.reset_index(drop=True, inplace=True)

df_c = df_ciro_filter.compare(df_antonio_in_final)
df_c.to_csv('Compare_v4_by_Ciro_vs_Antonio_in_final-3.csv')

df_ciro_filter.to_csv('Glucose_measurements_v4_by_Ciro.csv')
df_antonio_in_final.to_csv('Glucose_measurements_antonio_in_final.csv')


#
# # utils/df_diff_view.py
# from __future__ import annotations
# from pathlib import Path
# from typing import Iterable, Literal, Optional, Tuple
#
# import pandas as pd
# import numpy as np
#
#
# def _align_inputs(
#     df1: pd.DataFrame,
#     df2: pd.DataFrame,
#     columns: Optional[Iterable[str]] = None,
#     join: Literal["inner", "outer", "left", "right"] = "inner",
# ) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
#     if join == "inner":
#         idx = df1.index.intersection(df2.index)
#     elif join == "outer":
#         idx = df1.index.union(df2.index)
#     elif join == "left":
#         idx = df1.index
#     else:
#         idx = df2.index
#
#     if columns is None:
#         if join in ("outer",):
#             cols = sorted(set(df1.columns) | set(df2.columns))
#         else:
#             cols = list(df1.columns.intersection(df2.columns))
#     else:
#         cols = list(columns)
#
#     df1a = df1.reindex(index=idx, columns=cols)
#     df2a = df2.reindex(index=idx, columns=cols)
#     return df1a, df2a, cols
#
#
# def _diff_mask(df1a: pd.DataFrame, df2a: pd.DataFrame) -> pd.DataFrame:
#     m = (df1a.ne(df2a)) | (df1a.isna() ^ df2a.isna())
#     return m.fillna(True)
#
#
# def _apply_numeric_format(  # MOD
#     styler: pd.io.formats.style.Styler,  # MOD
#     df: pd.DataFrame,                    # MOD
#     precision: int,                      # MOD
# ) -> pd.io.formats.style.Styler:         # MOD
#     fmt = f"{{:.{precision}g}}"          # MOD
#     num_cols = df.select_dtypes(include="number").columns  # MOD
#     return styler.format({col: fmt for col in num_cols}, na_rep="")  # MOD
#
#
# def _apply_mask_color(  # MOD
#     styler: pd.io.formats.style.Styler,  # MOD
#     mask: pd.DataFrame,                  # MOD
#     color: str = "#ffe1e1",              # MOD
# ) -> pd.io.formats.style.Styler:         # MOD
#     """Aplica color celda-a-celda usando .apply por columna."""  # MOD
#     def _style_col(col: pd.Series) -> list[str]:  # MOD
#         m = mask[col.name].reindex(index=col.index)  # MOD
#         return [f"background-color:{color}" if bool(v) else "" for v in m]  # MOD
#     return styler.apply(_style_col, axis=0)  # MOD
#
#
# def side_by_side_view(
#     df1: pd.DataFrame,
#     df2: pd.DataFrame,
#     *,
#     columns: Optional[Iterable[str]] = None,
#     rows: Optional[Iterable] = None,
#     join: Literal["inner", "outer", "left", "right"] = "inner",
#     only_diffs: bool = True,
#     precision: int = 6,
# ) -> pd.io.formats.style.Styler:
#     df1a, df2a, cols = _align_inputs(df1, df2, columns=columns, join=join)
#     if rows is not None:
#         df1a = df1a.loc[rows]
#         df2a = df2a.loc[rows]
#
#     mask = _diff_mask(df1a, df2a)
#
#     wide = pd.concat({"df1": df1a, "df2": df2a}, axis=1)
#     interleaved_cols = []
#     for c in cols:
#         interleaved_cols.extend([("df1", c), ("df2", c)])
#     wide = wide.reindex(columns=pd.MultiIndex.from_tuples(interleaved_cols))
#
#     mask_wide = pd.concat({"df1": mask, "df2": mask}, axis=1)
#     mask_wide = mask_wide.reindex(columns=wide.columns)
#
#     if only_diffs:
#         row_keep = mask.any(axis=1)
#         wide = wide.loc[row_keep]
#         mask_wide = mask_wide.loc[row_keep]
#
#     styler = (
#         wide.style
#         .set_properties(**{"white-space": "nowrap"})
#         .set_table_styles(
#             [
#                 {"selector": "th.col_heading", "props": [("border-bottom", "1px solid #999")]},
#                 {"selector": "th.row_heading", "props": [("border-right", "1px solid #999")]},
#             ]
#         )
#         # .highlight_where(mask_wide, props="background-color:#ffe1e1;")  # MOD (eliminado)
#     )
#     styler = _apply_mask_color(styler, mask_wide, "#ffe1e1")  # MOD
#     styler = _apply_numeric_format(styler, wide, precision)   # MOD
#
#     friendly_cols = [f"{c} (df1)" if lvl0 == "df1" else f"{c} (df2)" for lvl0, c in wide.columns]
#     styler = styler.set_table_attributes('class="side-by-side"').set_caption("Comparación side-by-side")
#     styler.columns = friendly_cols
#     return styler
#
#
# def stacked_view(
#     df1: pd.DataFrame,
#     df2: pd.DataFrame,
#     *,
#     columns: Optional[Iterable[str]] = None,
#     rows: Optional[Iterable] = None,
#     join: Literal["inner", "outer", "left", "right"] = "inner",
#     only_diffs: bool = True,
#     precision: int = 6,
# ) -> pd.io.formats.style.Styler:
#     df1a, df2a, cols = _align_inputs(df1, df2, columns=columns, join=join)
#     if rows is not None:
#         df1a = df1a.loc[rows]
#         df2a = df2a.loc[rows]
#
#     mask = _diff_mask(df1a, df2a)
#     if only_diffs:
#         row_keep = mask.any(axis=1)
#         df1a = df1a.loc[row_keep]
#         df2a = df2a.loc[row_keep]
#         mask = mask.loc[row_keep]
#
#     long = pd.concat({"df1": df1a, "df2": df2a}, names=["source"])
#     mask_long = pd.concat({"df1": mask, "df2": mask}, names=["source"])
#     mask_long = mask_long.reindex(index=long.index, columns=long.columns).fillna(True)
#
#     styler = (
#         long.style
#         .set_properties(**{"white-space": "nowrap"})
#         .set_table_styles(
#             [
#                 {"selector": "th.row_heading", "props": [("border-right", "1px solid #999")]},
#             ]
#         )
#         # .highlight_where(mask_long, props="background-color:#fff0b3;")  # MOD (eliminado)
#     )
#     styler = _apply_mask_color(styler, mask_long, "#fff0b3")  # MOD
#     styler = _apply_numeric_format(styler, long, precision)   # MOD
#     styler = styler.set_table_attributes('class="stacked"').set_caption("Comparación stacked (df1 encima de df2 por índice)")
#     return styler
#
#
# def export_html(styler: pd.io.formats.style.Styler, path: Path | str) -> Path:
#     path = Path(path)
#     html = styler.to_html()
#     path.write_text(html, encoding="utf-8")
#     return path
#
#
# # ---- Ejemplos de uso (tu script) ------------------------------------------
# sbs = side_by_side_view(df_ciro_filter, df_antonio_in_final, only_diffs=True, precision=3)
# stk = stacked_view(df_ciro_filter, df_antonio_in_final, only_diffs=True, precision=3)
#
# export_html(sbs, "compare_side_by_side.html")
# export_html(stk, "compare_stacked.html")
# print("Generado: compare_side_by_side.html, compare_stacked.html")
