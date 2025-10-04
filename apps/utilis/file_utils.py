"""
File Utilities module - Handles file operations for the pipeline
"""
import json
import os
from typing import Dict, List, Any, Union


def save_json(data: Union[Dict, List], filepath: str) -> None:
    """
    Save data to a JSON file
    
    Args:
        data: Data to save
        filepath: Path to the output file
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Data saved to {filepath}")


def load_json(filepath: str) -> Union[Dict, List]:
    """
    Load data from a JSON file
    
    Args:
        filepath: Path to the input file
        
    Returns:
        The loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_document(filepath: str) -> Dict[str, str]:
    """
    Load a document from a file
    
    Args:
        filepath: Path to the document
        
    Returns:
        Dictionary with document content
    """
    # Extract document ID from the filename
    doc_id = os.path.splitext(os.path.basename(filepath))[0]

    # Determine document type and load accordingly
    if filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'id': doc_id,
            'content': content,
            'source': filepath
        }
    elif filepath.endswith('.json'):
        return load_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def load_documents(directory: str) -> List[Dict[str, str]]:
    """
    Load all documents from a directory
    
    Args:
        directory: Path to the directory containing documents
        
    Returns:
        List of document dictionaries
    """
    documents = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if os.path.isfile(filepath) and (filepath.endswith('.txt') or filepath.endswith('.json')):
            documents.append(load_document(filepath))

    return documents
