"""AFLOW materials database search via AFLUX REST API."""

import requests
from typing import Optional, Dict, Any, List

BASE_URL = "https://aflow.org/API/aflux/"

# Default fields to return in search results
DEFAULT_FIELDS = [
    "compound", "auid", "aurl", "species", "nspecies",
    "Egap", "energy_atom", "enthalpy_formation_atom",
    "spacegroup_relax", "lattice_system_relax",
    "volume_cell", "density",
    "Pearson_symbol_relax",
]


def _build_aflux_query(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    num_elements_min: Optional[int] = None,
    num_elements_max: Optional[int] = None,
    stability_min: Optional[float] = None,
    stability_max: Optional[float] = None,
    extra_fields: Optional[List[str]] = None,
) -> List[str]:
    """Build AFLUX query parts list."""
    parts = []

    # Select fields
    fields = list(DEFAULT_FIELDS)
    if extra_fields:
        for f in extra_fields:
            if f not in fields:
                fields.append(f)
    parts.extend(fields)

    # Element filter: species(Li,O) matches materials containing ALL listed elements
    if elements:
        parts.append(f"species({','.join(elements)})")

    # Number of species filter
    if num_elements_min is not None and num_elements_max is not None:
        if num_elements_min == num_elements_max:
            parts.append(f"nspecies({num_elements_min})")
        else:
            parts.append(f"nspecies({num_elements_min}*,*{num_elements_max})")
    elif num_elements_min is not None:
        parts.append(f"nspecies({num_elements_min}*)")
    elif num_elements_max is not None:
        parts.append(f"nspecies(*{num_elements_max})")

    # Band gap filter
    if band_gap_min is not None and band_gap_max is not None:
        parts.append(f"Egap({band_gap_min}*,*{band_gap_max})")
    elif band_gap_min is not None:
        parts.append(f"Egap({band_gap_min}*)")
    elif band_gap_max is not None:
        parts.append(f"Egap(*{band_gap_max})")

    # Stability filter (enthalpy_formation_atom)
    if stability_min is not None and stability_max is not None:
        parts.append(f"enthalpy_formation_atom({stability_min}*,*{stability_max})")
    elif stability_min is not None:
        parts.append(f"enthalpy_formation_atom({stability_min}*)")
    elif stability_max is not None:
        parts.append(f"enthalpy_formation_atom(*{stability_max})")

    return parts


def search_aflow(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    stability_min: Optional[float] = None,
    stability_max: Optional[float] = None,
    num_elements_min: Optional[int] = None,
    num_elements_max: Optional[int] = None,
    limit: int = 20,
    extra_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search the AFLOW materials database via AFLUX REST API.

    Args:
        elements: List of elements the material must contain, e.g. ["Li", "O"]
        band_gap_min: Minimum band gap in eV
        band_gap_max: Maximum band gap in eV
        stability_min: Minimum formation enthalpy per atom (eV/atom)
        stability_max: Maximum formation enthalpy per atom (eV/atom)
        num_elements_min: Minimum number of element species
        num_elements_max: Maximum number of element species
        limit: Maximum number of results to return
        extra_fields: Additional AFLUX property names to include

    Returns:
        Dict with "data" key containing list of material dicts
    """
    try:
        parts = _build_aflux_query(
            elements=elements,
            band_gap_min=band_gap_min,
            band_gap_max=band_gap_max,
            num_elements_min=num_elements_min,
            num_elements_max=num_elements_max,
            stability_min=stability_min,
            stability_max=stability_max,
            extra_fields=extra_fields,
        )

        # paging(1,limit) = first page with 'limit' results per page
        query_str = ",".join(parts) + f",paging(1,{limit})"
        url = BASE_URL + "?" + query_str
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        text = resp.text.strip()
        if not text:
            return {"data": [], "meta": {"total": 0}}

        data = resp.json()

        # Paginated response is a dict keyed by "N of TOTAL"
        if isinstance(data, dict):
            materials = []
            total = 0
            for key, value in data.items():
                if isinstance(value, dict):
                    materials.append(value)
                    if not total and " of " in key:
                        total = int(key.split(" of ")[-1])
            return {"data": materials, "meta": {"total": total or len(materials)}}
        elif isinstance(data, list):
            return {"data": data, "meta": {"total": len(data)}}
        else:
            return {"data": [], "meta": {"total": 0}}

    except requests.exceptions.RequestException as e:
        return {"error": f"AFLOW request failed: {str(e)}", "data": []}
    except Exception as e:
        return {"error": f"AFLOW search error: {str(e)}", "data": []}


def get_structure_data(auid: str, aurl: str) -> Dict[str, Any]:
    """Retrieve detailed structure data for an AFLOW material.

    Uses the AFLOWLIB server to fetch geometry via the aurl path.
    """
    try:
        # The aurl has format: "aflowlib.duke.edu:AFLOWDATA/..."
        # Convert to REST API URL: https://aflowlib.duke.edu/AFLOWDATA/.../?geometry
        path = aurl.split(":", 1)[1] if ":" in aurl else aurl
        url = f"https://{aurl.split(':', 1)[0] if ':' in aurl else 'aflowlib.duke.edu'}/{path}/"
        resp = requests.get(url + "?geometry", timeout=30)
        resp.raise_for_status()
        return {"success": True, "data": resp.json(), "auid": auid}
    except Exception as e:
        return {"success": False, "error": str(e), "auid": auid}
