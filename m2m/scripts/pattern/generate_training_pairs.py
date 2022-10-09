LANGS="af ar bg bn de el es et fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh".split()

print("Pairs: {}".format(len(LANGS)))
cmds = []
for lang in LANGS:
    cmd = "en-{}".format(lang)
    cmds.append(cmd)

print(",".join(cmds))
