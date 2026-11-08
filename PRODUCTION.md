# Qualified production build

- Runtime source commit: `c1592dc1fb97776057e4f68852cadce346596808`
- Runtime source tree: `d1868e4f3be4cec47636f8dd23089a42ace39c02`
- Image: `tensorfold-dsv41:image-yield-reviewed-c1592dc1`
- Image patch label: `1334c3cd624a`
- Checkpoint revision: `02d2ab346129098f9b3546143bb69944d88c7f977218627b916128cb9c1edfc9`
- Required switch: `TF_DS_PREFILL_YIELD=1`

Verified on all four production ranks:

- `tensorfold-native`: `d5bce398e58953e2711b96a7ad83cd66759fcedf22712a6816de033b6bf5e9f4`
- `tf-dsv41-lanes`: `041120365d322e34dd456b0fc117a7eba34a9ed5c993e233bf572b89ccf8321a`

The rollback image is `tensorfold-dsv41:native-6b3711e64978`.
