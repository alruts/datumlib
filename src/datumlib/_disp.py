from dataclasses import fields, is_dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from datumlib._containers import Datum, DatumCollection

console = Console()


def smart_truncate_repr(obj, max_chars=120, max_lines=4):
    try:
        r = repr(obj)
    except Exception:
        return f"<unrepresentable {type(obj).__name__}>"

    lines = r.splitlines()

    # Truncate lines
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["…"]

    r = "\n".join(lines)

    # Truncate chars
    if len(r) > max_chars:
        r = r[:max_chars] + " ..."

    return r


def display_datum(datum: Datum) -> None:
    """Display a Datum using only emphasis (bold/italic) for structure."""
    tree = Tree(f"[bold]{datum.__class__.__name__}[/bold]", guide_style="dim")

    # Data
    tree.add(
        f"[italic]data[/italic]: [bold]{smart_truncate_repr(datum.data)}[/bold]",
        expanded=False,
    )

    # Subclass fields
    if is_dataclass(datum):
        for field_ in fields(datum):
            if field_.name in ("data", "tags"):
                continue
            value = getattr(datum, field_.name)
            tree.add(f"[italic]{field_.name}[/italic]: [bold]{value!r}[/bold]")

    # Tags
    tags_tree = tree.add("[italic]tags[/italic]")
    if getattr(datum, "tags", None):
        for key, value in datum.tags.items():
            tags_tree.add(f"[bold]{key}[/bold]: {value!r}")
    else:
        tags_tree.add("[dim]No tags[/dim]")

    console.print(tree)


def display_collection(collection: DatumCollection) -> None:
    """Display a DatumCollection using only emphasis (bold/italic)."""
    # Collection tags
    if getattr(collection, "tags", None):
        tags_table = Table(show_header=False, box=None, pad_edge=False)
        for key, value in collection.tags.items():
            tags_table.add_row(f"[bold]{key}[/bold]", f"{value}")
    else:
        tags_table = "[dim]No collection tags[/dim]"

    # Entries tree
    entries_tree = Tree(
        f"[italic]entries[/italic] ([bold]{len(getattr(collection, 'entries', []))}[/bold])"
    )
    entries = getattr(collection, "entries", [])
    if entries:
        for i, entry in enumerate(entries):
            if entry is None:
                entries_tree.add(f"[dim]Entry {i}: None[/dim]")
                continue

            entry_tree = entries_tree.add(
                f"[bold]{i} {entry.__class__.__name__}[/bold]"
            )
            # Data
            entry_tree.add(
                f"[italic]data[/italic]: [bold]{smart_truncate_repr(entry.data)}[/bold]",
                expanded=False,
            )

            if is_dataclass(entry):
                for field_ in fields(entry):
                    if field_.name in ("data", "tags"):
                        continue
                    value = getattr(entry, field_.name)
                    entry_tree.add(
                        f"[italic]{field_.name}[/italic]: [bold]{value!r}[/bold]"
                    )

            # Entry tags
            entry_tags_tree = entry_tree.add("[italic]tags[/italic]")
            if getattr(entry, "tags", None):
                for key, value in entry.tags.items():
                    entry_tags_tree.add(f"[bold]{key}[/bold]: {value!r}")
            else:
                entry_tags_tree.add("[dim]No tags[/dim]")
    else:
        entries_tree.add("[dim]No entries[/dim]")

    # Build panel
    panel_content = Table.grid(padding=(0, 1))
    panel_content.add_row("[bold]Collection Tags[/bold]")
    panel_content.add_row(tags_table)
    panel_content.add_row(entries_tree)

    panel = Panel(
        panel_content,
        title=f"[bold]{collection.__class__.__name__}[/bold]",
        border_style="dim",
    )
    console.print(panel)
