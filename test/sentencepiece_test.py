import sentencepiece as spm

# 训练数据文件路径
input_file = 'tiny_file.txt'  # 请确保此文件存在并包含你的训练文本
model_prefix = 'spm_model'  # 模型前缀
vocab_size = 15  # 词汇表大小

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(input=input_file,
                               model_prefix=model_prefix,
                               model_type='bpe',
                               vocab_size=vocab_size,
                               character_coverage=1.0,
                               # num_threads=4,
                               split_digits=True,
                               allow_whitespace_only_pieces=True,
                               # byte_fallback=True,
                               unk_surface=r" \342\201\207 ",
                               normalization_rule_name="identity"
                               )

# 加载训练好的模型
sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')

# 输出每个分词及其概率分数（log-probability）
print("分词及其分数:")
for id in range(sp.get_piece_size()):
    piece = sp.id_to_piece(id)
    score = sp.get_score(id)  # 获取每个分词的分数
    print(f"分词: {piece}, 分数: {score}")
