#!/bin/bash
# The Developer ID certificate, on a runner: a keychain of its own made
# for the length of the job, the certificate imported into it from the
# MACOS_CERTIFICATE secret (the .p12, base64) and its password, and the
# keychain put on the search list, so build_macos.sh finds the identity
# exactly as it does on the desk.
#
#     packaging/ci_signing.sh
#
# The keychain's own password is random and never leaves this script;
# the partition list is what lets codesign use the key without the
# prompt nobody is there to answer. Apple's Developer ID intermediate
# goes in beside it, because an exported .p12 carries the leaf alone and
# codesign must build the chain to Apple's root. The runner is thrown
# away after the job, keychain and all.
set -euo pipefail
: "${MACOS_CERTIFICATE:?the .p12, base64}" "${MACOS_CERTIFICATE_PASSWORD:?its password}"
keychain="$RUNNER_TEMP/signing.keychain-db"
password=$(openssl rand -hex 24)
security create-keychain -p "$password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$password" "$keychain"
p12="$RUNNER_TEMP/certificate.p12"
trap 'rm -f "$p12" "$RUNNER_TEMP/DeveloperIDG2CA.cer"' EXIT
printf '%s' "$MACOS_CERTIFICATE" | base64 --decode > "$p12"
security import "$p12" -k "$keychain" -P "$MACOS_CERTIFICATE_PASSWORD" \
    -T /usr/bin/codesign -T /usr/bin/security
curl -sSL -o "$RUNNER_TEMP/DeveloperIDG2CA.cer" \
    https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer
security import "$RUNNER_TEMP/DeveloperIDG2CA.cer" -k "$keychain" || true
security set-key-partition-list -S apple-tool:,apple:,codesign: \
    -s -k "$password" "$keychain" > /dev/null
security list-keychains -d user -s "$keychain" \
    $(security list-keychains -d user | tr -d '"')
security find-identity -v -p codesigning "$keychain"
