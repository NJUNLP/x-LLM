dataset=$1
m1=$2
m2=$3
f1="model/${m1}/test/${dataset}.inference.jsonl"
f2="model/${m2}/test/${dataset}.inference.jsonl"
eval_model=gpt-3.5-turbo # evaluaotr gpt-4 or gpt-3.5-turbo
bpc=0  # 0/1 whether use the BPC strategy
k=1 # the evidence number of MEC strategy

# edit your openai key in FairEval.py first
python3 eval/mieval.eval.py \
    -q ${f1} \
    -a ${f1} ${f2} \
    -o "eval/log/${m1}.VS.${m2}.${dataset}.${eval_model}.json" \
    -m $eval_model \
    --bpc $bpc \
    -k $k 
    