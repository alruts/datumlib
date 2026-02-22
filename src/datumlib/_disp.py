from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from dataclasses import fields, is_dataclass

from datumlib import Datum, DatumCollection

console = Console()


def display_datum(datum: Datum) -> None:
    """
    Print a tree-style visualization of a Datum object, including subclass fields,
    with a soft aesthetic color scheme.
    """
    tree = Tree(f"[#c792ea]{datum.__class__.__name__}[/#c792ea]", guide_style="#9da5b4")

    # Display data
    tree.add(f"[#89ddff]data[/#89ddff]: [#ffcb6b]{datum.data!r}[/#ffcb6b]")

    # Display subclass-specific fields (skip 'data' and 'tags')
    if is_dataclass(datum):
        for field_ in fields(datum):
            if field_.name in ("data", "tags"):
                continue
            value = getattr(datum, field_.name)
            tree.add(f"[#89ddff]{field_.name}[/#89ddff]: [#ffcb6b]{value!r}[/#ffcb6b]")

    # Display tags
    tags_tree = tree.add("[#89ddff]tags[/#89ddff]")
    if getattr(datum, "tags", None):
        for key, value in datum.tags.items():
            tags_tree.add(f"[#f07178]{key}[/#f07178]: [#ffcb6b]{value!r}[/#ffcb6b]")
    else:
        tags_tree.add("[#7f848e]No tags[/#7f848e]")

    console.print(tree)


def display_collection(collection: DatumCollection) -> None:
    """
    Print a DatumCollection object with collection tags as a list
    and entries as a tree, wrapped in a rounded panel.
    """

    if collection.tags:
        tags_table = Table(show_header=False, box=None, pad_edge=False)
        for key, value in collection.tags.items():
            tags_table.add_row(
                f"[#f07178]{key}[/#f07178]", f"[#ffcb6b]{value}[/#ffcb6b]"
            )
    else:
        tags_table = "[#7f848e]No collection tags[/#7f848e]"

    entries_tree = Tree(
        f"[#89ddff]entries[/#89ddff] ([#ffcb6b]{len(collection.entries)}[/#ffcb6b])"
    )
    if collection.entries:
        for i, entry in enumerate(collection.entries):
            if entry is None:
                entries_tree.add(f"[#7f848e]Entry {i}: None[/#7f848e]")
                continue

            datum_tree = entries_tree.add(f"[bold #c792ea]Datum {i}[/bold #c792ea]")
            datum_tree.add(
                f"[#89ddff]data[/#89ddff]: [#ffcb6b]{entry.data!r}[/#ffcb6b]"
            )

            datum_tags_tree = datum_tree.add("[#89ddff]tags[/#89ddff]")
            if entry.tags:
                for key, value in entry.tags.items():
                    datum_tags_tree.add(
                        f"[#f07178]{key}[/#f07178]: [#ffcb6b]{value}[/#ffcb6b]"
                    )
            else:
                datum_tags_tree.add("[#7f848e]No tags[/#7f848e]")
    else:
        entries_tree.add("[#7f848e]No entries[/#7f848e]")

    panel_content = Table.grid(padding=(0, 1))
    panel_content.add_row("[bold green]Collection Tags[/bold green]")
    panel_content.add_row(tags_table)
    panel_content.add_row(entries_tree)

    panel = Panel(
        panel_content,
        title="[bold cyan]DatumCollection[/bold cyan]",
        border_style="bright_blue",
    )
    console.print(panel)
