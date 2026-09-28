# Security Policy

## Supported versions

PixForge is pre-1.0 and the only released version is the latest tag on the
[releases page](https://github.com/krubka1/pixforge/releases). Fixes land on
`main`; there are no long-term support branches.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private reporting on the Security tab
([Security → Report a vulnerability](https://github.com/krubka1/pixforge/security/advisories/new)).
That reaches the maintainer privately and is the only reporting channel
currently offered.

Please include:

- What the issue is and which file or feature it touches.
- How to reproduce it, ideally with a `.gltf`/`.glb` or `.pixforge` file that
  triggers it.
- What you expected and what happened instead.

## What matters most here

PixForge reads untrusted files by design — `.gltf`/`.glb` models, `.pixforge`
projects, GIMP `.gbr` brushes, image masks, palettes and `.hdr` environments.
Malformed input reaching a panic, an out-of-bounds read or an infinite loop is
the main risk surface, and the `.pixforge` reader is the part most worth
probing: it parses a custom little-endian binary format and has to reject
truncated and hostile input.

A crash in the renderer, a wrong-looking export or a brush that does not load
are bugs worth reporting, but not security issues.

## Signed binaries

The MSI is released unsigned. See the Installing section of the README.
