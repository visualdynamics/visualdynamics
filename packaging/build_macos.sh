#!/usr/bin/env bash
# Visual Dynamics.app, and a .dmg around it.
#
#     packaging/build_macos.sh [--sign "Developer ID Application: ..."]
#
# Signed and notarized when the machine can: a "Developer ID
# Application" identity in the keychain is used without being asked
# for (or name one with --sign, or VD_SIGN), and a notarytool keychain
# profile (NOTARY_PROFILE, default vd-notary) sends the app and then
# the image to Apple, waits, and staples the tickets — so the download
# opens with no right-click anywhere. Without the identity the bundle
# is ad-hoc signed and Gatekeeper asks once; without the profile it is
# signed but not notarized, and the build says so. Brandon joined the
# Apple Developer Program 2026-09-02; packaging/README.md has the two
# one-time steps (the certificate, the stored credentials).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-./.venv/bin/python}
VERSION=$("$PYTHON" -c "import visualdynamics; print(visualdynamics.__version__)")
APP="dist/Visual Dynamics.app"
# the architecture is the *interpreter's*, not the machine's: the
# Intel dmg is cross-built here by an x86_64 Python under Rosetta
# (build_macos_intel.sh), and uname would name it arm64
ARCH=$("$PYTHON" -c "import platform; print(platform.machine())")
DMG="dist/VisualDynamics-${VERSION}-macos-${ARCH}.dmg"
SIGN=${VD_SIGN:-}
[[ ${1:-} == --sign ]] && SIGN=${2:?identity required}
if [[ -z $SIGN ]]; then
  # the one kind of identity Gatekeeper accepts for a download; an
  # "Apple Development" certificate is for running on your own Macs
  SIGN=$(security find-identity -v -p codesigning 2>/dev/null |
         sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)
fi
# How notarytool is told who we are, in order of preference:
#
# 1. A login-keychain item named NOTARY_ITEM (default vd-notary-password)
#    holding the app-specific password, with the Apple ID as its
#    account — made once with `security add-generic-password` (see
#    packaging/README.md, "Signing"). An ordinary keychain item, so a
#    script reads it without a prompt for as long as the login keychain
#    is unlocked, which it is whenever Brandon is logged in.
# 2. The notarytool profile NOTARY_PROFILE (default vd-notary), which
#    `notarytool store-credentials` keeps in the *data-protection*
#    keychain under the com.apple.gke.notary access group. macOS opens
#    that only to an unlocked user session: from an agent's shell it
#    answered "No Keychain password item found" for hours at a time and
#    came back the moment Brandon touched it from his own Terminal
#    (2026-09-12 to 09-15), which is why the item above exists.
NOTARY_PROFILE=${NOTARY_PROFILE:-vd-notary}
NOTARY_ITEM=${NOTARY_ITEM:-vd-notary-password}
NOTARY_TEAM=${NOTARY_TEAM:-2542NQ9D95}
NOTARY=()
notary_args() {
  local account password
  if password=$(security find-generic-password -s "$NOTARY_ITEM" -w 2>/dev/null) \
      && [[ -n $password ]]; then
    account=$(security find-generic-password -s "$NOTARY_ITEM" 2>/dev/null |
              sed -n 's/.*"acct"<blob>="\(.*\)"/\1/p')
    NOTARY=(--apple-id "$account" --team-id "$NOTARY_TEAM" --password "$password")
    echo "notarizing as $account (login-keychain item $NOTARY_ITEM)"
  else
    NOTARY=(--keychain-profile "$NOTARY_PROFILE")
  fi
}
notary_ready() {
  # asking for the submission history is the one check that proves the
  # credentials still work
  [[ -n $SIGN ]] || return 1
  notary_args
  xcrun notarytool history "${NOTARY[@]}" >/dev/null 2>&1
}

rm -rf build dist
"$PYTHON" -m PyInstaller packaging/visualdynamics.spec --noconfirm \
  --distpath dist --workpath build

if [[ -n $SIGN ]]; then
  # --deep is deprecated and unreliable for nested code; sign inside
  # out. Every Mach-O *by content*, not by name: the first notarized
  # build signed *.dylib and *.so and Apple refused it 125 times over,
  # once per Qt framework binary (no extension) and once for the
  # WebEngine helper app, all still carrying Qt's own signature
  # (2026-09-02). Files deepest first, then the nested bundles — the
  # frameworks and the helper app, which needs the hardened runtime
  # like the main executable — then the app itself.
  sign_flags=(--force --timestamp --options runtime
              --entitlements packaging/entitlements.plist --sign "$SIGN")
  find "$APP/Contents" -type f -print0 | xargs -0 file |
    grep -E ': *Mach-O' | cut -d: -f1 |
    awk '{print gsub("/", "/") "\t" $0}' | sort -rn | cut -f2- |
    tr '\n' '\0' | xargs -0 -n 40 codesign "${sign_flags[@]}" 2>&1 |
    grep -v 'replacing existing signature' || true
  find "$APP/Contents" \( -name '*.framework' -o -name '*.app' \) -depth -print0 |
    xargs -0 -n 1 codesign "${sign_flags[@]}" 2>&1 |
    grep -v 'replacing existing signature' || true
  codesign "${sign_flags[@]}" "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | tail -2
  echo "signed as $SIGN"
  if notary_ready; then
    # the app first, on its own, so its ticket can be stapled *into*
    # the bundle before the image is made — a ticket stapled only to
    # the image leaves the copied app asking Apple on first launch,
    # which fails offline
    ZIP="dist/notarize-$$.zip"
    ditto -c -k --keepParent "$APP" "$ZIP"
    xcrun notarytool submit "$ZIP" "${NOTARY[@]}" --wait | tail -3
    rm -f "$ZIP"
    xcrun stapler staple -q "$APP"
    spctl --assess --type execute -v "$APP"
  else
    echo "warning: signed but not notarized — neither the login-keychain" \
         "item '$NOTARY_ITEM' nor the notarytool profile '$NOTARY_PROFILE'" \
         "answers (packaging/README.md, Signing)" >&2
  fi
else
  # An ad-hoc signature, which is not a real one: it makes the bundle
  # launchable on arm64, where an *unsigned* binary is refused outright
  # rather than merely warned about.
  codesign --force --deep --sign - "$APP" 2>/dev/null || true
fi

# The window a person opens: the app on the left, /Applications on the
# right, an arrow between — drag across, installed. The layout is a
# .DS_Store Finder writes into a read-write image, which is then
# compressed; headless (a runner without a Finder session), the layout
# step is skipped and the image is the plain folder it always was.
rm -f "$DMG"
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
mkdir "$STAGE/.background"
"$PYTHON" packaging/dmg_background.py "$STAGE/.background/background.png"

RW="dist/rw-$$.dmg"
hdiutil create -volname "Visual Dynamics ${VERSION}" -srcfolder "$STAGE" \
  -ov -format UDRW "$RW" >/dev/null
rm -rf "$STAGE"
MOUNT=$(hdiutil attach -readwrite -noverify -noautoopen "$RW" |
        awk -F'\t' '/\/Volumes\//{print $3}')
if osascript >/dev/null 2>&1 <<OSA
tell application "Finder"
  tell disk "$(basename "$MOUNT")"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {200, 120, 860, 568}
    set opts to the icon view options of container window
    set arrangement of opts to not arranged
    set icon size of opts to 128
    set background picture of opts to file ".background:background.png"
    set position of item "Visual Dynamics.app" to {165, 195}
    set position of item "Applications" to {495, 195}
    close
  end tell
end tell
OSA
then :; else
  echo "warning: no Finder session — the image is unarranged" >&2
fi
sync
hdiutil detach "$MOUNT" -quiet || { sleep 2; hdiutil detach "$MOUNT" -force -quiet; }
hdiutil convert "$RW" -format UDZO -o "$DMG" -quiet
rm -f "$RW"
if [[ -n $SIGN ]]; then
  # the image itself, then its own ticket: what Gatekeeper sees first
  # is the disk image, and an unsigned one around a notarized app
  # still draws the warning
  codesign --force --timestamp --sign "$SIGN" "$DMG"
  if notary_ready; then
    xcrun notarytool submit "$DMG" "${NOTARY[@]}" --wait | tail -3
    xcrun stapler staple -q "$DMG"
    spctl --assess --type open --context context:primary-signature -v "$DMG"
    echo "notarized and stapled"
  fi
fi
echo "built $DMG ($(du -sh "$DMG" | cut -f1))"
