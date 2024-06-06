from jiwer import AbstractTransform
import unidecode

class Normalize(AbstractTransform):
    """
    Normalizes the string to ASCII.

    This transform converts any accented or special characters to their closest ASCII equivalents,
    without altering leading or trailing spaces.

    Example:
        ```python
        import jiwer

        sentences = [" this is an example ", "  héllo goodbÿe  ", "  "]

        print(jiwer.Normalize()(sentences))
        # prints: [' this is an example ', "  hello goodbye  ", "  "]
        # note that leading and trailing spaces are preserved
        ```
    """

    def process_string(self, s: str):
        """
        Process the input string by normalizing it to ASCII.

        Args:
            s (str): The input string to be processed.

        Returns:
            str: The processed string with all characters converted to their closest ASCII equivalents.
        """
        # Normalize the string to ASCII
        return unidecode.unidecode(s)