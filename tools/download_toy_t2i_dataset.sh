target_dir="datasets/t2i_toy_dataset"
mkdir -p $target_dir
base_url="https://huggingface.co/datasets/GoodEnough/TiM-Toy-T2I-Dataset/resolve/main"
files=(
    "bucket_categorization.json"
    "bucket_sampler.json"
    "data_info.jsonl"
    "dataset.tar.gz"
    "get_sampler.py"
)
for file in "${files[@]}"; do
    echo "download $file ..."
    wget -c "$base_url/$file" -O "$target_dir/$file"
    echo "download $file finished"
done
tar -xzvf $target_dir/dataset.tar.gz -C $target_dir
echo "Successfully download all the dataset"

