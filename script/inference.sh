MODEL_NAME=$1
DATASET=$2
BATCH=${3:-8}

CPFS_PATH=/home/user
PROJECT_PATH=$CPFS_PATH/project/mt_instruction_tuning

export HF_HOME=$CPFS_PATH/.cache/huggingface

GEN_ARGS=()
case $DATASET in  
	mmlu*)
		GEN_ARGS+=("--template raw")
		GEN_ARGS+=("--labels A B C D")		
		GEN_ARGS+=("--max_new_tokens 1")
		# GEN_ARGS+=("--evaluate perplexity")
		;;
	belebele*zeroshot)
		GEN_ARGS+=("--labels A B C D")
		GEN_ARGS+=("--max_new_tokens 1")
		# GEN_ARGS+=("--evaluate perplexity")
		;;
	xcoparaw*)
		GEN_ARGS+=("--template raw")
		GEN_ARGS+=("--labels A B")		
		GEN_ARGS+=("--evaluate perplexity")
		;;
	xcopa*)
		GEN_ARGS+=("--labels A B")		
		GEN_ARGS+=("--evaluate perplexity")
		;;
	ceval*)
		GEN_ARGS+=("--template raw")
		GEN_ARGS+=("--labels A B C D")			
		GEN_ARGS+=("--max_new_tokens 1")
		# GEN_ARGS+=("--evaluate perplexity")
		;;
	xwinograd*)
		GEN_ARGS+=("--evaluate perplexity")
		;;
	xnli*)
		GEN_ARGS+=("--labels entailment neutral contradiction")		
		GEN_ARGS+=("--evaluate perplexity")
		;;
	pawsx*)
		GEN_ARGS+=("--labels yes no")
		GEN_ARGS+=("--evaluate perplexity")
		;;
	*)  
		;;
esac


case $MODEL_NAME in  
	llama-2-7b-chat-hf)
		GEN_ARGS+=("--template raw")
		;; 
	bloom-7b1*)
		GEN_ARGS+=("--load_in_8bit True")
		;;
	*)  
		;;
esac

source $CPFS_PATH/miniconda3/bin/activate $PROJECT_PATH/.env

mkdir -p "$PROJECT_PATH/model/$MODEL_NAME/test"

python \
    $PROJECT_PATH/inference.py \
    --data_path "$PROJECT_PATH/data/$DATASET" \
    --model_name_or_path "$PROJECT_PATH/model/$MODEL_NAME" \
	${GEN_ARGS[@]} \
    --batch_size $BATCH \
    --output_file "$PROJECT_PATH/model/$MODEL_NAME/test/$DATASET.inference.jsonl"
    