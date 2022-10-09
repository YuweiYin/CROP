LANGS="af ar bg bn de el en es et fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh".split()
cmds = ""


for lang in LANGS:
    cmd = """
- name: filter_{}
  sku: G0
  sku_count: 1
  command: 
    - bash ./shells/pattern/train-data/filter_lang.sh {}""".format(lang, lang)
    cmds += cmd

print(cmds)
