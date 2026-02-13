from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from monolib.containers import Datum, DatumCollection

console = Console()


def display_mono(m: Datum):
    """
    Print a tree-style visualization of a Datum object with a soft aesthetic color scheme.
    """
    tree = Tree("[#c792ea]Datum[/#c792ea]", guide_style="#9da5b4")

    # Data field
    tree.add(f"[#89ddff]data[/#89ddff]: [#ffcb6b]{m.data.__repr__()}[/#ffcb6b]")

    # Tags
    tags_tree = tree.add("[#89ddff]tags[/#89ddff]")
    if m.tags:
        for key, value in m.tags.items():
            tags_tree.add(f"[#f07178]{key}[/#f07178]: [#ffcb6b]{value}[/#ffcb6b]")
    else:
        tags_tree.add("[#7f848e]No tags[/#7f848e]")

    console.print(tree)


def display_collection(mc: "DatumCollection"):
    """
    Print a DatumCollection object with collection tags as a list
    and entries as a tree, wrapped in a rounded panel.
    """

    # Collection tags as a table/list
    if mc.tags:
        tags_table = Table(show_header=False, box=None, pad_edge=False)
        for key, value in mc.tags.items():
            tags_table.add_row(
                f"[#f07178]{key}[/#f07178]", f"[#ffcb6b]{value}[/#ffcb6b]"
            )
    else:
        tags_table = "[#7f848e]No collection tags[/#7f848e]"

    # Entries as a tree
    entries_tree = Tree(
        f"[#89ddff]entries[/#89ddff] ([#ffcb6b]{len(mc.entries)}[/#ffcb6b])"
    )
    if mc.entries:
        for i, entry in enumerate(mc.entries):
            if entry is None:
                entries_tree.add(f"[#7f848e]Entry {i}: None[/#7f848e]")
                continue

            mono_tree = entries_tree.add(f"[bold #c792ea]Datum {i}[/bold #c792ea]")
            mono_tree.add(f"[#89ddff]data[/#89ddff]: [#ffcb6b]{entry.data!r}[/#ffcb6b]")

            mono_tags_tree = mono_tree.add("[#89ddff]tags[/#89ddff]")
            if entry.tags:
                for key, value in entry.tags.items():
                    mono_tags_tree.add(
                        f"[#f07178]{key}[/#f07178]: [#ffcb6b]{value}[/#ffcb6b]"
                    )
            else:
                mono_tags_tree.add("[#7f848e]No tags[/#7f848e]")
    else:
        entries_tree.add("[#7f848e]No entries[/#7f848e]")

    # Combine tags and tree into a single panel
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
