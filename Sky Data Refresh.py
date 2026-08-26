from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd
import streamlit as st


SKY_REQUIRED_COLUMNS = {
    "Account ID",
    "Name",
    "Billing Street",
    "Billing City",
    "Post Code",
    "Billing County",
    "Country",
}

SITE_UPLOAD_COLUMNS = [
    "name",
    "address_1",
    "address_2",
    "address_3",
    "city",
    "post_code",
    "county",
    "country",
    "code",
    "status",
    "client_internal_id",
    "client_org_level_1",
    "client_org_level_2",
    "client_org_level_3",
    "client_org_level_4",
]

GB_COUNTRIES = {
    "ENGLAND",
    "ENGLAND & WALES",
    "GREAT BRITAIN",
    "SCOTLAND",
    "WALES",
}

IE_COUNTRIES = {
    "IRELAND",
    "NORTHERN IRELAND",
    "REPUBLIC OF IRELAND",
    "ROI",
}


class InputFileError(ValueError):
    """A user-facing problem with an uploaded file."""


def cell_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def read_csv_bytes(data: bytes, label: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            frame = pd.read_csv(
                io.BytesIO(data),
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
            frame.columns = [cell_text(column) for column in frame.columns]
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            raise InputFileError(f"{label} could not be read as a CSV: {exc}") from exc
    raise InputFileError(f"{label} could not be decoded as UTF-8 or Windows-1252.") from last_error


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise InputFileError(f"{label} is missing required column(s): {', '.join(missing)}")


def read_excel_table(
    data: bytes,
    required: set[str],
    label: str,
    preferred_sheet: str | None = None,
) -> pd.DataFrame:
    try:
        book = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except Exception as exc:
        raise InputFileError(f"{label} could not be opened as an Excel workbook: {exc}") from exc

    sheet_names = list(book.sheet_names)
    if preferred_sheet in sheet_names:
        sheet_names.remove(preferred_sheet)
        sheet_names.insert(0, preferred_sheet)

    for sheet_name in sheet_names:
        preview = pd.read_excel(
            book,
            sheet_name=sheet_name,
            header=None,
            nrows=10,
            dtype=object,
        )
        for row_number, row in preview.iterrows():
            labels = {cell_text(value) for value in row.tolist() if cell_text(value)}
            if required.issubset(labels):
                frame = pd.read_excel(
                    book,
                    sheet_name=sheet_name,
                    header=int(row_number),
                    dtype=object,
                )
                frame.columns = [cell_text(column) for column in frame.columns]
                return frame

    raise InputFileError(
        f"{label} does not contain a sheet with the required headers: "
        f"{', '.join(sorted(required))}"
    )


def load_sky_data(data: bytes) -> pd.DataFrame:
    frame = read_excel_table(
        data,
        SKY_REQUIRED_COLUMNS,
        "Sky data file",
        preferred_sheet="Sheet1",
    )
    require_columns(frame, SKY_REQUIRED_COLUMNS, "Sky data file")
    frame = frame.copy()
    frame["Account ID"] = frame["Account ID"].map(cell_text)
    frame = frame.loc[(frame["Account ID"] != "") & (frame["Account ID"] != "Account ID")]
    return frame.reset_index(drop=True)


def load_audits_export(data: bytes) -> pd.DataFrame:
    frame = read_csv_bytes(data, "Audits export")
    require_columns(frame, {"internal_id", "site_code"}, "Audits export")
    frame = frame.copy()
    frame["internal_id"] = frame["internal_id"].map(cell_text)
    frame["site_code"] = frame["site_code"].map(cell_text)
    return frame


def compare_site_codes(
    sky: pd.DataFrame, audits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sky_codes = set(sky.loc[sky["Account ID"] != "", "Account ID"])
    audit_codes = set(audits.loc[audits["site_code"] != "", "site_code"])

    removals = audits.loc[
        (audits["site_code"] != "") & (~audits["site_code"].isin(sky_codes))
    ].copy()
    additions = sky.loc[
        (sky["Account ID"] != "") & (~sky["Account ID"].isin(audit_codes))
    ].copy()
    additions = additions.drop_duplicates(subset=["Account ID"], keep="first")
    return removals.reset_index(drop=True), additions.reset_index(drop=True)


def dataframe_to_csv_bytes(frame: pd.DataFrame, bom: bool = False) -> bytes:
    def csv_value(value: object) -> str:
        text_value = "" if value is None or pd.isna(value) else str(value)
        if any(character in text_value for character in (",", '"', "\r", "\n", "\t")):
            return '"' + text_value.replace('"', '""') + '"'
        return text_value

    rows = [",".join(csv_value(column) for column in frame.columns)]
    rows.extend(
        ",".join(csv_value(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    )
    text = "\r\n".join(rows) + "\r\n"
    return text.encode("utf-8-sig" if bom else "utf-8")


def build_audit_delete_template(removals: pd.DataFrame) -> pd.DataFrame:
    result = removals.loc[:, ["internal_id"]].copy()
    result["status"] = "deleted"
    return result


def load_sites_export(data: bytes, label: str = "Sky sites export") -> pd.DataFrame:
    frame = read_csv_bytes(data, label)
    require_columns(frame, {"internal_id", "code"}, label)
    frame = frame.copy()
    frame["internal_id"] = frame["internal_id"].map(cell_text)
    frame["code"] = frame["code"].map(cell_text)
    return frame


def missing_addition_sites(additions: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    site_codes = set(sites.loc[sites["code"] != "", "code"])
    return additions.loc[~additions["Account ID"].isin(site_codes)].reset_index(drop=True)


def split_street(value: object) -> tuple[str, str, str]:
    street = cell_text(value)
    if not street:
        return "", "", ""
    parts = [part.strip() for part in street.split(",", maxsplit=2)]
    parts += [""] * (3 - len(parts))
    return parts[0], parts[1], parts[2]


def system_country(value: object) -> str:
    country = cell_text(value).upper()
    return "Ireland" if country in {"IRELAND", "REPUBLIC OF IRELAND", "ROI"} else "United Kingdom"


def build_sites_upload_template(missing_sites: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, site in missing_sites.iterrows():
        address_1, address_2, address_3 = split_street(site.get("Billing Street", ""))
        rows.append(
            {
                "name": cell_text(site.get("Name", "")),
                "address_1": address_1,
                "address_2": address_2,
                "address_3": address_3,
                "city": cell_text(site.get("Billing City", "")),
                "post_code": cell_text(site.get("Post Code", "")),
                "county": cell_text(site.get("Billing County", "")),
                "country": system_country(site.get("Country", "")),
                "code": cell_text(site.get("Account ID", "")),
                "status": "active",
                "client_internal_id": "CLIENT152",
                "client_org_level_1": "company",
                "client_org_level_2": "",
                "client_org_level_3": "",
                "client_org_level_4": "",
            }
        )
    return pd.DataFrame(rows, columns=SITE_UPLOAD_COLUMNS)


def load_token_map(data: bytes) -> dict[str, str]:
    frame = read_excel_table(
        data,
        {"PC", "MC Region"},
        "NARV and MC Patches workbook",
        preferred_sheet="Overall",
    )
    require_columns(frame, {"PC", "MC Region"}, "NARV and MC Patches workbook")

    token_map: dict[str, str] = {}
    for _, row in frame.iterrows():
        prefix = re.sub(r"\s+", "", cell_text(row["PC"]).upper())
        token = cell_text(row["MC Region"])
        if prefix and token and not token.startswith("="):
            token_map.setdefault(prefix, token)

    if not token_map:
        raise InputFileError(
            "No usable MC Region values were found on the Overall sheet. "
            "Open the tokens workbook in Excel, recalculate and save it, then upload it again."
        )
    return token_map


def token_for_site(site: pd.Series, token_map: dict[str, str]) -> str:
    post_code = re.sub(r"\s+", "", cell_text(site.get("Post Code", "")).upper())
    matches = [prefix for prefix in token_map if post_code.startswith(prefix)]
    if matches:
        return token_map[max(matches, key=len)]

    country = cell_text(site.get("Country", "")).upper()
    if country in IE_COUNTRIES:
        return "MC Ireland"
    return ""


def country_group(value: object) -> str:
    country = cell_text(value).upper()
    if country in GB_COUNTRIES:
        return "GB"
    if country in IE_COUNTRIES:
        return "IE"
    return ""


def build_additions_files(
    additions: pd.DataFrame,
    sites: pd.DataFrame,
    token_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_id_by_code: dict[str, str] = {}
    for _, row in sites.iterrows():
        code = cell_text(row["code"])
        internal_id = cell_text(row["internal_id"])
        if code and internal_id and code not in site_id_by_code:
            site_id_by_code[code] = internal_id

    gb_rows: list[dict[str, str]] = []
    ie_rows: list[dict[str, str]] = []
    missing_site_ids: list[str] = []
    missing_tokens: list[str] = []
    unknown_countries: list[str] = []

    for _, site in additions.iterrows():
        code = cell_text(site["Account ID"])
        site_internal_id = site_id_by_code.get(code, "")
        token = token_for_site(site, token_map)
        group = country_group(site.get("Country", ""))

        if not site_internal_id:
            missing_site_ids.append(code)
        if not token:
            missing_tokens.append(code)
        if not group:
            unknown_countries.append(f"{code} ({cell_text(site.get('Country', '')) or 'blank'})")

        row = {"site_internal_id": site_internal_id, "tokens": token}
        if group == "GB":
            gb_rows.append(row)
        elif group == "IE":
            ie_rows.append(row)

    problems: list[str] = []
    if missing_site_ids:
        problems.append("sites not found in the sites export: " + ", ".join(missing_site_ids))
    if missing_tokens:
        problems.append("sites with no MC token mapping: " + ", ".join(missing_tokens))
    if unknown_countries:
        problems.append("sites with an unrecognised country: " + ", ".join(unknown_countries))
    if problems:
        raise InputFileError("Cannot create the additions files because there are " + "; ".join(problems) + ".")

    columns = ["site_internal_id", "tokens"]
    return (
        pd.DataFrame(gb_rows, columns=columns),
        pd.DataFrame(ie_rows, columns=columns),
    )


def compact_preview(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available]


@st.cache_data(show_spinner=False)
def cached_initial_analysis(sky_bytes: bytes, audit_bytes: bytes):
    sky = load_sky_data(sky_bytes)
    audits = load_audits_export(audit_bytes)
    removals, additions = compare_site_codes(sky, audits)
    return sky, audits, removals, additions


@st.cache_data(show_spinner=False)
def cached_sites_export(data: bytes, label: str) -> pd.DataFrame:
    return load_sites_export(data, label)


@st.cache_data(show_spinner=False)
def cached_token_map(data: bytes) -> dict[str, str]:
    return load_token_map(data)


def render_download(label: str, data: bytes, filename: str, key: str) -> None:
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Sky Data Refresh", page_icon="🔄", layout="wide")
    st.title("Sky Data Refresh")
    st.write(
        "Upload the latest Sky data and current audits export. Site-code matching is "
        "case-sensitive. The app will guide you through any additional files only when needed."
    )

    date_tag = date.today().strftime("%d-%m")

    left, right = st.columns(2)
    with left:
        sky_upload = st.file_uploader(
            "1. Upload Sky data file",
            type=["xlsx"],
            key="sky_data",
            help="For example: Serve Legal UK ROI Combined Data 250826.xlsx",
        )
    with right:
        audits_upload = st.file_uploader(
            "2. Upload audits export",
            type=["csv"],
            key="audits_export",
            help="For example: audits_benchmarker_export.csv",
        )

    if sky_upload is None or audits_upload is None:
        st.info("Upload both files to begin the comparison.")
        return

    try:
        with st.spinner("Comparing case-sensitive site codes…"):
            sky, audits, removals, additions = cached_initial_analysis(
                sky_upload.getvalue(), audits_upload.getvalue()
            )
    except InputFileError as exc:
        st.error(str(exc))
        return

    blank_sky_codes = int((sky["Account ID"] == "").sum())
    blank_audit_codes = int((audits["site_code"] == "").sum())
    if blank_sky_codes or blank_audit_codes:
        st.warning(
            f"Blank codes were ignored: {blank_sky_codes} in the Sky file and "
            f"{blank_audit_codes} in the audits export."
        )

    duplicate_sky_codes = int(sky["Account ID"].duplicated(keep=False).sum())
    if duplicate_sky_codes:
        st.warning(
            f"The Sky file contains {duplicate_sky_codes} rows whose Account ID is duplicated. "
            "Each duplicated code is treated as one site for additions."
        )

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Sky sites", f"{sky['Account ID'].nunique():,}")
    metric_b.metric("Removals", f"{len(removals):,}")
    metric_c.metric("Additions", f"{len(additions):,}")

    st.subheader("Removals")
    if removals.empty:
        st.success("No audits need to be removed.")
    else:
        with st.expander(f"View {len(removals):,} removal(s)"):
            st.dataframe(
                compact_preview(
                    removals,
                    ["internal_id", "site_internal_id", "site_code", "site_name", "site_post_code"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        removal_col, delete_col = st.columns(2)
        with removal_col:
            render_download(
                f"Download Removals {date_tag}.csv",
                dataframe_to_csv_bytes(removals, bom=True),
                f"Removals {date_tag}.csv",
                "download_removals",
            )
            st.caption("Send this file to Ops to confirm the removals.")
        with delete_col:
            render_download(
                "Download audits_upload_template.csv",
                dataframe_to_csv_bytes(build_audit_delete_template(removals)),
                "audits_upload_template.csv",
                "download_audit_deletions",
            )
            st.caption("Upload this file to remove the audits from the system.")

    st.subheader("Additions")
    if additions.empty:
        st.success("No sites need to be added. The refresh files are complete.")
        return

    with st.expander(f"View {len(additions):,} addition(s)"):
        st.dataframe(
            compact_preview(
                additions,
                ["Account ID", "Name", "Billing City", "Post Code", "Country"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Additions were found. Upload the current Sky sites export and the tokens workbook "
        "to continue."
    )
    site_col, token_col = st.columns(2)
    with site_col:
        sites_upload = st.file_uploader(
            "3. Upload Sky sites export",
            type=["csv"],
            key="sites_export",
            help="For example: sites_code_only_export.csv",
        )
    with token_col:
        tokens_upload = st.file_uploader(
            "4. Upload NARV and MC Patches workbook",
            type=["xlsx"],
            key="tokens_workbook",
            help="The MC Region values are read from the Overall sheet.",
        )

    if sites_upload is None or tokens_upload is None:
        return

    try:
        sites = cached_sites_export(sites_upload.getvalue(), "Sky sites export")
        token_map = cached_token_map(tokens_upload.getvalue())
    except InputFileError as exc:
        st.error(str(exc))
        return

    missing_sites = missing_addition_sites(additions, sites)
    if not missing_sites.empty:
        st.warning(
            f"{len(missing_sites):,} addition(s) do not yet exist in the sites export. "
            "Create those sites first, then upload a new sites export below."
        )
        with st.expander(f"View {len(missing_sites):,} site(s) to create"):
            st.dataframe(
                compact_preview(
                    missing_sites,
                    ["Account ID", "Name", "Billing City", "Post Code", "Country"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        render_download(
            "Download sites_upload_template.csv",
            dataframe_to_csv_bytes(build_sites_upload_template(missing_sites)),
            "sites_upload_template.csv",
            "download_sites_template",
        )

        updated_sites_upload = st.file_uploader(
            "5. Upload the updated Sky sites export after creating the sites",
            type=["csv"],
            key="updated_sites_export",
        )
        if updated_sites_upload is None:
            return

        try:
            sites = cached_sites_export(updated_sites_upload.getvalue(), "Updated Sky sites export")
        except InputFileError as exc:
            st.error(str(exc))
            return

        remaining_missing = missing_addition_sites(additions, sites)
        if not remaining_missing.empty:
            st.error(
                f"The updated sites export is still missing {len(remaining_missing):,} required "
                "site(s). Create them and upload another updated export."
            )
            with st.expander("View sites still missing"):
                st.dataframe(
                    compact_preview(
                        remaining_missing,
                        ["Account ID", "Name", "Billing City", "Post Code", "Country"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            return

        st.success("All required sites are now present in the updated sites export.")

    try:
        gb_additions, ie_additions = build_additions_files(additions, sites, token_map)
    except InputFileError as exc:
        st.error(str(exc))
        return

    if gb_additions.empty and ie_additions.empty:
        st.error("No additions output could be created. Check the Country values in the Sky file.")
        return

    st.success("The additions files are ready.")
    output_columns = st.columns(2)
    if not gb_additions.empty:
        with output_columns[0]:
            render_download(
                f"Download GB Additions {date_tag}.csv",
                dataframe_to_csv_bytes(gb_additions),
                f"GB Additions {date_tag}.csv",
                "download_gb_additions",
            )
            st.caption("England, Scotland and Wales")
    if not ie_additions.empty:
        with output_columns[1]:
            render_download(
                f"Download IE Additions {date_tag}.csv",
                dataframe_to_csv_bytes(ie_additions),
                f"IE Additions {date_tag}.csv",
                "download_ie_additions",
            )
            st.caption("Northern Ireland and the Republic of Ireland")


if __name__ == "__main__":
    main()
