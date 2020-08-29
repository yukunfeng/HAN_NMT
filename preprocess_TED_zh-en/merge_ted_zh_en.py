def merge_ted_zh_en():
  years = ["2010", "2011", "2012", "2013"]
  doc_files = [f"./zh-en-ted-extracted/IWSLT15.TED.tst{year}.zh-en.doc" for year in years]
  src_files = [f"./zh-en-ted-preprocessed/IWSLT15.TED.tst{year}.tc.zh" for year in years]
  tgt_files = [f"./zh-en-ted-preprocessed/IWSLT15.TED.tst{year}.tc.en" for year in years]

  merged_src_file = open("./zh-en-ted-preprocessed/IWSLT15.TED.tst2010_2013.tc.zh", "w")
  merged_tgt_file = open("./zh-en-ted-preprocessed/IWSLT15.TED.tst2010_2013.tc.en", "w")
  merged_doc_file = open("./zh-en-ted-extracted/IWSLT15.TED.tst2010_2013.zh-en.doc", "w")

  previous_len = 0
  for doc_file, src_file, tgt_file in zip(doc_files, src_files, tgt_files):
    tmp_docs = []
    with open(doc_file, 'r') as fh:
      for line in fh:
        line = line.strip()
        if line == "":
          continue
        tmp_docs.append(int(line))

    tmp_src = []
    with open(src_file, 'r') as fh:
      for line in fh:
        tmp_src.append(line)

    tmp_tgt = []
    with open(tgt_file, 'r') as fh:
      for line in fh:
        tmp_tgt.append(line)
    if len(tmp_tgt) != len(tmp_src):
      raise Exception("src and tgt: #sent are same")
    
    # Update current doc
    for i, doc in enumerate(tmp_docs, 0):
      tmp_docs[i] += previous_len

    previous_len += len(tmp_src)

    # Write
    for doc in tmp_docs:
      merged_doc_file.write(f"{doc}\n")
    for sent in tmp_src:
      merged_src_file.write(f"{sent}")
    for sent in tmp_tgt:
      merged_tgt_file.write(f"{sent}")

  merged_src_file.close()
  merged_tgt_file.close()
  merged_doc_file.close()


if __name__ == "__main__":
  merge_ted_zh_en()
