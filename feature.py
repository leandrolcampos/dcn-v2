from collections import defaultdict

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Text,
    TypeVar,
)


T = TypeVar('T')


class Feature(Generic[T]):
    def __init__(
        self,
        name: Text,
        original_name: Optional[Text] = None,
        transform_fn: Optional[Callable[[Any], T]] = None,
    ) -> None:
        self._name = name
        self._original_name = original_name or name
        self._transform_fn = transform_fn or (lambda x: x)

    def __call__(self, x: Any) -> T:
        return self._transform_fn(x)

    @property
    def name(self) -> Text:
        return self._name

    @property
    def original_name(self) -> Text:
        return self._original_name


class ContinuousFeature(Feature[float]):
    pass


class IntegerFeature(Feature[int]):
    pass


class StringFeature(Feature[Text]):
    pass


def build_features_dict(features: List[Feature]) -> Dict[Text, Feature]:
    features_dict = {}
    label_count = 0

    for feature in features:
        if not isinstance(feature, Feature):
            raise TypeError(
                'Every item in features must be an instance of Feature.'
            )
        name = feature.name
        label_count = label_count + 1 if name == 'label' else label_count
        features_dict[name] = feature

    if label_count != 1:
        raise ValueError(
            f"Expected one feature named 'label' but got {label_count}."
        )

    return features_dict


def features_by_type(
    features_dict: Dict[Text, Feature]
) -> Dict[Text, List[Text]]:
    types = {
        ContinuousFeature: 'continuous',
        IntegerFeature: 'integer',
        StringFeature: 'string',
    }

    features = defaultdict(list)
    for name, feature in features_dict.items():
        features[types[type(feature)]].append(name)

    return features
