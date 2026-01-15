import argparse
from .image_serialize import ImageSerializer
from .data_clean import DataCleaner
import json
import os
import sys

SUBTYPE_DEFAULT_CLASSES = [
    "SV_NONE",
    "SV_FIRE_ENGINE",
    "SV_AMBULANCE",
    "SV_POLICE_CAR",
    "SV_SPRINKLER",
    "SV_SPRINKLER_WO_BRUSH",
    "SV_SANITIZATION_VEHICLE",
    "SV_SCHOOL_BUS",
    "STOP_SIGN_UNSEEN",
    "STOP_SIGN_ON",
    "STOP_SIGN_OFF",
    "PROFESSION_NONE",
    "PROFESSION_POLICE",
    "PROFESSION_CONSTRUCTOR",
]

def main():
    parser = argparse.ArgumentParser(description='Semantic Clean CLI Tool')
    parser.add_argument('--generate', action='store_true', help='Generate database from JSON file')
    parser.add_argument('--clean', action='store_true', help='Clean target data using database')
    parser.add_argument('--file', type=str, help='Input JSON file path for generation')
    parser.add_argument('--base', type=str, help='Base database path for reference')
    parser.add_argument('--target', type=str, help='Target JSON file to clean')
    parser.add_argument('--output', type=str, help='Output JSON file for cleaned results')
    parser.add_argument('--db-path', type=str, default='chroma_db', help='Path to ChromaDB database')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing (default: 50)')
    
    args = parser.parse_args()
    
    try:
        if args.generate and args.file:
            if not os.path.exists(args.file):
                print(f"Error: Input file {args.file} does not exist")
                return 1
                
            serializer = ImageSerializer(db_path=args.db_path)
            print(f"Generating database from {args.file}...")
            serializer.load_from_json(args.file)
            print(f"Database generated at {args.db_path}")
            
        elif args.clean and args.base and args.target and args.output:
            if not os.path.exists(args.base):
                print(f"Error: Base database path {args.base} does not exist")
                return 1
                
            if not os.path.exists(args.target):
                print(f"Error: Target file {args.target} does not exist")
                return 1
            
            print(f"Starting data cleaning process...")
            print(f"Base database: {args.base}")
            print(f"Target file: {args.target}")
            print(f"Output file: {args.output}")
            print(f"Batch size: {args.batch_size}")
            
            cleaner = DataCleaner(db_path=args.base,
                                  validate_categories=SUBTYPE_DEFAULT_CLASSES)
            results = cleaner.clean_target_data(args.target, args.output, batch_size=args.batch_size)
            
            # 显示统计信息
            stats = cleaner.get_statistics(results)
            print("\n=== 清洗结果统计 ===")
            print(f"总计: {stats['total']}")
            print(f"接受 (Accept): {stats['accept']} ({stats['accept_rate']:.2%})")
            print(f"拒绝 (Reject): {stats['reject']} ({stats['reject_rate']:.2%})")
            print(f"人工审核 (Review): {stats['review']} ({stats['review_rate']:.2%})")
            if stats['error'] > 0:
                print(f"处理错误: {stats['error']}")
            
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())