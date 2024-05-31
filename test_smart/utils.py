"""
This file defines a few convenient tools for use throughout the package.
"""

from enum import Enum


class AmbiguousItemQuery(ValueError):
    pass


class UnknownItem(ValueError):
    pass


class QueryEnum(Enum):
    """
    QueryEnum is an Enum which you can construct conveniently by writing a
    string query. For example, for an enum Fruit with items Fruit.APPLE and
    Fruit.BANANA, you may wish to construct via Fruit.from_str("apple"), or even
    Fruit.from_str("a")...
    """

    @classmethod
    def from_str(cls, query: str):
        query = query.lower()

        # Search for martingales which start with the query
        matches = {}
        for martingale in cls:
            name = martingale.name.lower()
            val = martingale.value.lower()
            if name.startswith(query) or val.startswith(query):
                matches[martingale] = matches.get(martingale, 0) + 1
        if len(matches) == 1:
            return list(matches.keys())[0]
        elif len(matches) > 1:
            matches = [key.value for key in matches.keys()]
            raise AmbiguousItemQuery(
                f"Ambiguous item query: {query}. Matches: {matches}."
            )
        else:
            raise UnknownItem(
                f"No item found matching {query}."
                f"Expected something resembling one of "
                f"{[item.value for item in cls]}."
            )
