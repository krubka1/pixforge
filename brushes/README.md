# Shipped brushes

These folders are loaded at runtime by the Brushes panel (`sw_scan` in
`src/brushes.rs`): each subfolder becomes a category and every `.gbr`/`.png`
inside is offered as a texture stamp. Drop your own files in here (or point the
`PIXFORGE_BRUSHES` env var elsewhere) to extend the library.

## Sources & licenses

| Category | Source | License |
| --- | --- | --- |
| `Spot`, `Stroke`, `Spiky`, `Grain` | [gimp-brush-collection](https://github.com/vascoalexander/gimp-brush-collection) by Vasco Alexander Basque | CC0 1.0 |
| `Material`, `Dots` | Procedurally generated at build time — `--gen-brush-packs` regenerates them from `src/brushes.rs` | Original, no third-party license |

All packs permit free use. The app's built-in procedural brushes (Round,
Square, Diamond, Splotch, Grain, Wood, Marble, Rust, Brushed Metal, Hammered
Metal, Halftone, Dot Grid, Checker, Diamond Plate, Canvas, Concrete, Grunge,
Pebbled Leather, Sand, Stone… much like the `Material`/`Dots` packs) are
generated deterministically in code and carry no third-party license. Run
`pixforge --gen-brush-packs [DIR]` to refresh the shipped folders.