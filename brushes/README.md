# Shipped brushes

These folders are loaded at runtime by the Brushes panel (`sw_scan` in
`src/brushes.rs`): each subfolder becomes a category and every `.gbr`/`.png`
inside is offered as a texture stamp. Drop your own files in here (or point the
`PIXFORGE_BRUSHES` env var elsewhere) to extend the library.

## Sources & licenses

| Category | Source | License |
| --- | --- | --- |
| `Spot`, `Stroke`, `Dots`, `Spiky`, `Grain` | [gimp-brush-collection](https://github.com/vascoalexander/gimp-brush-collection) by Vasco Alexander Basque | CC0 1.0 |

All packs permit free use. The app's built-in procedural brushes (Round,
Square, Diamond, Splotch, Grain, Wood, Marble, Rust…) are generated in code
and carry no third-party license.