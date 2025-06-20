import argparse
from modelscope import snapshot_download

def main():
    parser = argparse.ArgumentParser(description='Download model snapshots.')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='Name of the model to download (e.g., DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-32B)')
    parser.add_argument('--cache_dir', type=str, required=True, 
                        help='Directory to cache the downloaded model.')

    args = parser.parse_args()


    model_dir = snapshot_download(args.model_name, cache_dir=args.cache_dir, revision='master')
    print(f'Model downloaded to: {model_dir}')

if __name__ == '__main__':
    main()

