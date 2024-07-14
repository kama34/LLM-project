def write_to_file(file_path, content, clear_file=False):
    """
    Write content to a file. If clear_file is True, clear the file before writing.

    Parameters:
    file_path (str): The path to the file.
    content (str): The content to write to the file.
    clear_file (bool): If True, clear the file before writing. Default is False.
    """
    mode = 'w' if clear_file else 'a'

    with open(file_path, mode) as f:
        f.write(content + '\n')