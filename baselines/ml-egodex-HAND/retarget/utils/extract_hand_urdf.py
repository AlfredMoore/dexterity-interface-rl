"""
Extract a standalone Tesollo DG-3F hand URDF from the bimanual panda+tesollo URDF.
The extracted URDF contains only the hand links/joints with absolute mesh paths.

Usage:
    python extract_hand_urdf.py                  # generates both left and right
    python extract_hand_urdf.py --side right
    python extract_hand_urdf.py --side left --output /custom/path.urdf
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[4]  # .../dexterity-interface-rl/ (or /workspace/ in container)
_SOURCE_URDF = _REPO_ROOT / "libs/robot_description/rl/bimanual_panda_tesollo.urdf"
_OUTPUT_DIR = Path(__file__).parents[1] / "assets"


def extract_hand_urdf(side: str, output_path: Path, source_urdf: Path = _SOURCE_URDF) -> None:
    """
    Extract the Tesollo DG-3F hand for `side` ('left' or 'right') from the bimanual URDF.

    Included elements:
      Links:  {side}_delto_base_link, {side}_F[123]_01..04, {side}_F[123]_TIP
      Joints: {side}_F[123]M[1234] (revolute), {side}_TIP[123] (fixed)

    Mesh filenames are rewritten to absolute paths so the URDF is portable.
    """
    assert side in ("left", "right"), f"side must be 'left' or 'right', got {side!r}"
    source_dir = source_urdf.parent

    tree = ET.parse(source_urdf)
    src_root = tree.getroot()

    # All links/joints whose name starts with this prefix belong to the target hand.
    prefix = f"{side}_"
    hand_root_link = f"{side}_delto_base_link"

    # --- Build the new robot element ---
    new_robot = ET.Element("robot", attrib={"name": f"tesollo_dg3f_{side}"})

    def _fix_mesh_paths(element: ET.Element) -> None:
        """Replace relative mesh paths with absolute paths."""
        for mesh in element.findall(".//mesh"):
            filename = mesh.get("filename", "")
            if filename and not Path(filename).is_absolute():
                abs_path = (source_dir / filename).resolve()
                mesh.set("filename", str(abs_path))

    # Build parent→[child_link] map for the full URDF
    child_to_joint: dict[str, ET.Element] = {}
    parent_to_children: dict[str, list[str]] = {}
    for joint in src_root.findall("joint"):
        p = joint.find("parent")
        c = joint.find("child")
        if p is not None and c is not None:
            pname, cname = p.get("link"), c.get("link")
            parent_to_children.setdefault(pname, []).append(cname)
            child_to_joint[cname] = joint

    # BFS from hand_root_link to collect all reachable links/joints
    visited_links: set[str] = set()
    queue = [hand_root_link]
    while queue:
        link_name = queue.pop()
        if link_name in visited_links:
            continue
        visited_links.add(link_name)
        for child in parent_to_children.get(link_name, []):
            queue.append(child)

    # Add links in visited set
    link_by_name = {l.get("name"): l for l in src_root.findall("link")}
    for lname in visited_links:
        link_el = link_by_name.get(lname)
        if link_el is not None:
            _fix_mesh_paths(link_el)
            new_robot.append(link_el)

    # Add joints whose parent is in visited set
    for joint in src_root.findall("joint"):
        parent_el = joint.find("parent")
        if parent_el is not None and parent_el.get("link") in visited_links:
            new_robot.append(joint)

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(new_robot, space="  ")
    ET.ElementTree(new_robot).write(
        str(output_path), xml_declaration=True, encoding="unicode"
    )

    links  = [e.get("name") for e in new_robot.findall("link")]
    joints = [e.get("name") for e in new_robot.findall("joint")]
    print(f"Written: {output_path}")
    print(f"  Links  ({len(links)}):  {links}")
    print(f"  Joints ({len(joints)}): {joints}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract standalone Tesollo DG-3F URDF")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (only valid when --side is left or right)")
    args = parser.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]
    for side in sides:
        if args.output and len(sides) == 1:
            out = Path(args.output)
        else:
            out = _OUTPUT_DIR / f"tesollo_dg3f_{side}.urdf"
        extract_hand_urdf(side, out)


if __name__ == "__main__":
    main()
