import nussl

# TODO: Hack for now, PR into nussl.
nussl.separation.base.deep_mixin.OMITTED_TRANSFORMS = list(nussl.separation.base.deep_mixin.OMITTED_TRANSFORMS)
nussl.separation.base.deep_mixin.OMITTED_TRANSFORMS.append(
    nussl.datasets.transforms.IndexSources
)
nussl.separation.base.deep_mixin.OMITTED_TRANSFORMS = tuple(nussl.separation.base.deep_mixin.OMITTED_TRANSFORMS)
