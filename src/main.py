import argparse
from .image_searilize import ImageSerializer
import json
import os

def main():
    parser = argparse.ArgumentParser(description='Semantic Clean CLI Tool')
    parser.add_argument('--generate', action='store_true', help='Generate database from JSON file')
    parser.add_argument('--file', type=str, help='Input JSON file path')
    parser.add_argument('--db-path', type=str, default='chroma_db', help='Path to ChromaDB database')
    
    args = parser.parse_args()
    
    if args.generate and args.file:
        if not os.path.exists(args.file):
            print(f"Error: Input file {args.file} does not exist")
            return 1
            
        serializer = ImageSerializer(db_path=args.db_path)
        print(f"Generating database from {args.file}...")
        serializer.load_from_json(args.file)
        print(f"Database generated at {args.db_path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()