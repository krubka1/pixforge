//! Color palette library: named collections of RGBA swatches with a plain-text
//! import/export story. Reads GIMP palettes (`.gpl`), JASC-PAL (`.pal`) and
//! simple `R G B` / `#RRGGBB[AA]` text files, and writes GIMP `.gpl`.
//! Colors are stored as straight (premultiplied-free) sRGB RGBA bytes.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Palette {
    pub name: String,
    /// RGBA swatches.
    pub colors: Vec<[u8; 4]>,
}

impl Palette {
    pub fn from_presets(presets: &[[u8; 3]], name: &str) -> Self {
        Palette {
            name: name.to_string(),
            colors: presets
                .iter()
                .map(|c| [c[0], c[1], c[2], 255])
                .collect(),
        }
    }

    /// Serialize as a GIMP `.gpl` palette (the closest thing to a standard,
    /// plain-text color palette format).
    pub fn to_gpl(&self) -> String {
        let mut out = String::new();
        out.push_str("GIMP Palette\n");
        out.push_str(&format!("Name: {}\n", self.name));
        out.push_str("Columns: 8\n#\n");
        for c in &self.colors {
            out.push_str(&format!(
                "{:3} {:3} {:3}\t#{:02X}{:02X}{:02X}\n",
                c[0], c[1], c[2], c[0], c[1], c[2]
            ));
        }
        out
    }
}

/// Parse a palette from the bytes of a `.gpl`, `.pal` or bare `R G B` / hex
/// text file. `name_hint` is the fallback name (normally the file stem) when
/// the file carries no `Name:` field.
pub fn parse_palette(bytes: &[u8], name_hint: &str) -> Palette {
    let text = String::from_utf8_lossy(bytes);
    let mut name = name_hint.to_string();
    let mut colors: Vec<[u8; 4]> = Vec::new();

    for line in text.lines() {
        let raw = line.trim();
        if raw.is_empty() {
            continue;
        }
        // Structural headers that carry no color data.
        if raw == "JASC-PAL"
            || raw == "0100"
            || raw == "GIMP Palette"
            || raw.starts_with("Columns:")
            || raw.starts_with('!')
            || raw.starts_with("PAL")
        {
            continue;
        }
        if let Some(rest) = strip_prefix_ci(raw, "Name:") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                name = trimmed.to_string();
            }
            continue;
        }

        // An in-row hex swatch (GIMP appends `#RRGGBB` to some rows) wins over
        // the leading RGB triple, which may be repeatedly-rounded float values.
        if let Some(h) = raw
            .split_whitespace()
            .find(|t| t.starts_with('#') && hex_to_rgba(t).is_some())
        {
            colors.push(hex_to_rgba(h).unwrap());
            continue;
        }
        // A bare 6- or 8-hex-digit row (with or without '#') is a color — the
        // 6/8 length never collides with the small bare-integer bookkeeping.
        let bare = raw.strip_prefix('#').unwrap_or(raw);
        if (bare.len() == 6 || bare.len() == 8) && hex_to_rgba(bare).is_some() {
            colors.push(hex_to_rgba(bare).unwrap());
            continue;
        }
        // Anything else leading with '#' is a comment.
        if raw.starts_with('#') {
            continue;
        }
        // A bare integer is the JASC color count or similar bookkeeping.
        if raw.parse::<u64>().is_ok() {
            continue;
        }

        // "R G B [A]" rows (GIMP / JASC / plain text). A 4th integer, when
        // present, is alpha (0-255); GIMP/JASC-only rows default to opaque.
        let ints: Vec<u32> = raw
            .split_whitespace()
            .filter_map(|t| t.parse::<u32>().ok())
            .collect();
        if ints.len() >= 3 {
            let r = ints[0].min(255) as u8;
            let g = ints[1].min(255) as u8;
            let b = ints[2].min(255) as u8;
            let a = ints
                .get(3)
                .copied()
                .map(|a| a.min(255) as u8)
                .unwrap_or(255);
            colors.push([r, g, b, a]);
        }
    }

    Palette { name, colors }
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// Decode `#RRGGBB` / `#RRGGBBAA` (leading `#` optional).
fn hex_to_rgba(hex: &str) -> Option<[u8; 4]> {
    let digits = hex.strip_prefix('#').unwrap_or(hex);
    if digits.len() != 6 && digits.len() != 8 {
        return None;
    }
    let parse = |i: usize| u8::from_str_radix(&digits[i..i + 2], 16).ok();
    Some([
        parse(0)?,
        parse(2)?,
        parse(4)?,
        if digits.len() == 8 { parse(6)? } else { 255 },
    ])
}

fn hex_palette(name: &str, hexes: &[&str]) -> Palette {
    Palette {
        name: name.to_string(),
        colors: hexes.iter().filter_map(|h| hex_to_rgba(h)).collect(),
    }
}

/// Hand-picked swatches that seed the stock "PixForge Default" palette.
const PIXFORGE_PRESETS: [[u8; 3]; 12] = [
    [255, 255, 255], // white
    [0, 0, 0],       // black
    [90, 160, 255],  // pixforge blue
    [214, 65, 65],   // red
    [70, 160, 92],   // green
    [61, 139, 214],  // blue
    [224, 161, 60],  // amber
    [160, 90, 210],  // purple
    [214, 160, 160], // pink
    [90, 90, 90],    // gray
    [246, 241, 232], // cream
    [60, 200, 200],  // teal
];

/// Curated palettes shipped with the app so the library is useful out of the
/// box: the original quick-picks, a grayscale ramp (handy for mask/height
/// painting), a light-to-deep skin-tone ramp, and two popular pixel-art
/// palettes (DB16 by DawnBringer and PICO-8 by Lexaloffle).
pub fn builtin_palettes() -> Vec<Palette> {
    let grayscale: Vec<[u8; 4]> = (0..16)
        .map(|i| {
            let v = 255 - i * 17;
            [v, v, v, 255]
        })
        .collect();
    vec![
        Palette::from_presets(&PIXFORGE_PRESETS, "PixForge Default"),
        Palette {
            name: "Grayscale".to_string(),
            colors: grayscale,
        },
        hex_palette(
            "Skin Tones",
            &[
                "FFDBAC", "F1C27D", "E0AC69", "C68642", "8D5524", "6B3E1E", "4A2A14", "2E1A0C",
            ],
        ),
        hex_palette(
            "Retro 16 (DB16)",
            &[
                "140C1C", "442434", "30346D", "4E4A4E", "854C30", "346524", "D04648", "757161",
                "597DCE", "D27D2C", "8595A1", "6DAA2C", "D2AA99", "6DC2CA", "DAD45E", "DEEED6",
            ],
        ),
        hex_palette(
            "PICO-8",
            &[
                "000000", "1D2B53", "7E2553", "008751", "AB5236", "5F574F", "C2C3C7", "FFF1E8",
                "FF004D", "FFA300", "FFEC27", "00E436", "29ADFF", "83769C", "FF77A8", "FFCCAA",
                "291814", "111D35", "422136", "125359", "742F29", "49333B", "A28879", "F3EF7D",
                "BE1250", "FF6C24", "A8E72E", "00B543", "065AB5", "754665", "FF6E59", "FF9D81",
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpl_round_trip() {
        let pal = Palette {
            name: "Test".into(),
            colors: vec![[255, 0, 0, 255], [0, 128, 255, 255]],
        };
        let text = pal.to_gpl();
        let parsed = parse_palette(text.as_bytes(), "fallback");
        assert_eq!(parsed.name, "Test");
        assert_eq!(parsed.colors, pal.colors);
    }

    #[test]
    fn parses_jasc_pal() {
        let bytes = b"JASC-PAL\n0100\n2\n10 20 30\n40 50 60\n";
        let pal = parse_palette(bytes, "my");
        assert_eq!(pal.name, "my");
        assert_eq!(pal.colors, vec![[10, 20, 30, 255], [40, 50, 60, 255]]);
    }

    #[test]
    fn int_rows_keep_alpha() {
        // A trailing 4th value is alpha; without one the swatch stays opaque.
        let bytes = b"10 20 30 128\n40 50 60\n";
        let pal = parse_palette(bytes, "a");
        assert_eq!(pal.colors, vec![[10, 20, 30, 128], [40, 50, 60, 255]]);
    }

    #[test]
    fn parses_hex_rows() {
        let bytes = b"#ff0000\n#00ff0080\n112233\n";
        let pal = parse_palette(bytes, "h");
        assert_eq!(
            pal.colors,
            vec![[255, 0, 0, 255], [0, 255, 0, 128], [0x11, 0x22, 0x33, 255]]
        );
    }

    #[test]
    fn gimp_rows_with_trailing_hex_win() {
        // GIMP rows carry the hex swatch as the last token; the hex should win
        // over the leading RGB triple so rounded GIMP floats don't corrupt it.
        let bytes = b"GIMP Palette\nName: W\n255 0 0.5\t#FF007F\n";
        let pal = parse_palette(bytes, "w");
        assert_eq!(pal.name, "W");
        assert_eq!(pal.colors, vec![[255, 0, 0x7f, 255]]);
    }
}