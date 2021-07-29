import torch
import numpy as np
from typing import Tuple, Optional
import warnings
import pickle
import glob
import matplotlib.pyplot as plt
import random
import tqdm
from tqdm import tqdm
import os
import collections
from emoji import UNICODE_EMOJI
# from aai.alexandria.data.fetch.google import _download_page, fetch_google_images
import urllib
import urllib.request
from urllib.request import Request, urlopen
import json
import shutil

from references import raw_text_dir, visual_vocab_dir, npy_root_dir
from augment import augment_visual_vocab

warnings.filterwarnings("ignore")

stopwords = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", 
    "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", 
    "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", 
    "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", 
    "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", 
    "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways",
     "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", 
     "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", 
     "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", 
     "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", 
     "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", 
     "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", 
     "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", 
     "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", 
     "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", 
     "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", 
     "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely",
      "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", 
      "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", 
      "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", 
      "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", 
      "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", 
      "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", 
      "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", 
      "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", 
      "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", 
      "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't",
       "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello",
        "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", 
        "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", 
        "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", 
        "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", 
        "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead",
         "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", 
         "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", 
         "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", 
         "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", 
         "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", 
         "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely",
          "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", 
          "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", 
          "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", 
          "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", 
          "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", 
          "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", 
          "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", 
          "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", 
          "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", 
          "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", 
          "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", 
          "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", 
          "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", 
          "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", 
          "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", 
          "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", 
          "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", 
          "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", 
          "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", 
          "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", 
          "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", 
          "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", 
          "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", 
          "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", 
          "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", 
          "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", 
          "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", 
          "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", 
          "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", 
          "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", 
          "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", 
          "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", 
          "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", 
          "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", 
          "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's",
           "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", 
           "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", 
           "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", 
           "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", 
           "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",
           "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
           "@", "!", "#", "$", "%", "^", "&", "*", "(", ")", "_", "-", "+", "=", ":", "|", "/", "?", ";", ",", "'", ".",  "w/", "1st", "2nd", "3rd"]

def _download_page(url):
    headers = {}
    headers[
        'User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    req = urllib.request.Request(url, headers=headers)
    return str(urllib.request.urlopen(req).read())

def _get_next_item(s):
    start_line = s.find('rg_meta notranslate')
    if start_line == -1:  # If no links are found then give an error!
        return None
    start_line = s.find('class="rg_meta notranslate">')
    start_object = s.find('{', start_line + 1)
    end_object = s.find('</div>', start_object + 1)
    object_raw = str(s[start_object:end_object])
    # try:
    object_decode = bytes(object_raw, "utf-8").decode("unicode_escape")
    final_object = json.loads(object_decode)
    print(object_decode)
    print(final_object)
    # except UnicodeDecodeError:
    #     return None
    return final_object, end_object

def _get_all_items(page, image_limit):
    urls = []
    # Get the next image
    for _ in range(image_limit):
        img_obj, end_obj = _get_next_item(page)
        if img_obj is None:
            break
        page = page[end_obj:]
        urls.append(_format_object(img_obj))
    return urls

def fetch_google_images(keywords, image_limit=5):
    search_url = 'https://www.google.com/search?q={}&tbm=isch'.format('+'.join(urllib.parse.quote(k) for k in keywords))
    raw_html = _download_page(search_url)
    if raw_html is None:
        return []
    images = [img.replace("src=", "").split(';')[0][1:] for img in raw_html.split(" ") if "src=" in img and "https://" in img]
    return images[:image_limit]


def get_text_vocab(prefix='train', pos='verb', topk=1000):
    text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))
    word_counter = {}
    for k in tqdm(text_dict):
        words = text_dict[k].lower().split(" ")
        for w in words:
            w = w.replace("-", "").replace(".", "").replace("/", "").replace("_", "").replace("!", "").replace("(", "").replace(")", "").replace(":", "").replace(".", "").replace(",", "").replace(";", "").replace("#", "").replace("@", "").replace(" ", "")
            if w and  w != "video" and not w.isnumeric() and w not in UNICODE_EMOJI and  w not in stopwords:
                if (pos=='verb' and 'ing' in w) or (pos != 'verb' and 'ing' not in w):
                    if w not in word_counter:
                        word_counter[w] = 0
                    word_counter[w] += 1
    wordlist = [x[0] for x in sorted(word_counter.items(), key=lambda x: -x[1])]
    return wordlist[:topk]

def save_imgs(keywords):
    for k in keywords:
        try:
            images = fetch_google_images([k])
            for i, url in enumerate(images):
                dir = f"{visual_vocab_dir}/{k}"

                if not os.path.exists(dir):
                    os.makedirs(dir)

                urllib.request.urlretrieve(url, f'{dir}/{i}.jpg')
        except:
            continue

def remove_folders(keywords):
    for k in keywords:
        subdir = f'{visual_vocab_dir}/{k}'
        print(subdir)
        if os.path.exists(subdir):
            shutil.rmtree(subdir)


def create_visual_vocab():
    words = []
    words += get_text_vocab(pos='verb')
    words += get_text_vocab(pos='noun')

    print(len(words))

    save_imgs(words)

def get_visual_vocab_paths():
    path_imgs_dict = {}
    text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_train.pickle", "rb"))
    visual_vocab = set([subdir.split("/")[-1] for subdir in list(glob.glob(f'{visual_vocab_dir}/*'))])


    for path in tqdm(glob.glob(f"{npy_root_dir}/*/video/*")):
        url = path.split('/')[-1].split('.')[0]
        if url not in text_dict:
            continue
        text = text_dict[url].lower().split(" ")
        for word in text:
            if word in visual_vocab:
                if url not in path_imgs_dict:
                    path_imgs_dict[url] = []
                path_imgs_dict[url] += list(glob.glob(f'{visual_vocab_dir}/{word}/*'))

    print(len(path_imgs_dict))
    print(sum([len(vals) for vals in list(path_imgs_dict.values())]) / len(path_imgs_dict))
    with open('path_visual_vocab_dir.pickle', 'wb') as handle:
        pickle.dump(path_imgs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def probe_visual_vocab():
    path_imgs_dict = {}
    text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_train.pickle", "rb"))
    visual_vocab = pickle.load(open('path_visual_vocab_dir.pickle', "rb"))


    paths = list(glob.glob(f"{npy_root_dir}/*/video/*"))
    random.shuffle(paths)
    path = paths[0]

    url = path.split('/')[-1].split('.')[0]

    if url not in text_dict:
        return
    
    text = text_dict[url].lower().split(" ")
    print(text)

    if url in visual_vocab:
        img_paths = visual_vocab[url]
        random.shuffle(img_paths)
        imgs = [augment_visual_vocab(i).to(torch.int) for i in img_paths[:4]]
        f, arr = plt.subplots(2,2)
        arr[0,0].imshow(imgs[0])
        arr[0,1].imshow(imgs[1])
        arr[1,0].imshow(imgs[2])
        arr[1,1].imshow(imgs[3])
        plt.savefig('sanity_checks/visual_vocab_probe')

def get_visual_vocab_tag_positives(add, remove):
    tag_imgs_dict = {}
    img_tags_dict = {}
    text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_train.pickle", "rb"))
    visual_vocab = set([subdir.split("/")[-1] for subdir in list(glob.glob(f'{visual_vocab_dir}/*'))]+add)


    for path in tqdm(glob.glob(f"{npy_root_dir}/*/video/*")):
        url = path.split('/')[-1].split('.')[0]
        if url not in text_dict:
            continue
        text = text_dict[url].lower().split(" ")
        for word in text:
            alterations = [word, word[:-1], word+'s']
            for w in alterations:
                if w in visual_vocab and w not in remove:
                    if w not in tag_imgs_dict:
                        tag_imgs_dict[w] = set([])
                    tag_imgs_dict[w].add(path)

                    if url not in img_tags_dict:
                        img_tags_dict[url] = set([])
                    img_tags_dict[url].add(w)

    with open('tag_to_imgs.pickle', 'wb') as handle:
        pickle.dump(tag_imgs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('img_to_tags.pickle', 'wb') as handle:
        pickle.dump(img_tags_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def probe_tags_dict(query):
    tags_dict = pickle.load(open('tag_to_imgs.pickle', "rb"))
    tags = list(tags_dict[query])
    random.shuffle(tags)
    tags = [torch.from_numpy(np.load(tags[i])).to(torch.int) for i in range(4)]
    f, arr = plt.subplots(2,2)
    arr[0,0].imshow(tags[0][5])
    arr[0,1].imshow(tags[1][5])
    arr[1,0].imshow(tags[2][5])
    arr[1,1].imshow(tags[3][5])
    plt.savefig('sanity_checks/tags_dict')

        
if __name__ == "__main__":
    # topk_words = get_text_vocab()

    # for keyword in topk_words:
    #     images = fetch_google_images([keyword])
    #     for i, url in enumerate(images):
    #         dir = f"/big/sgurram/k700_visual_vocab/{keyword}"

    #         if not os.path.exists(dir):
    #             os.makedirs(dir)

    #         urllib.request.urlretrieve(url, f'{dir}/{i}.jpg')

    remove = ["youtube", "year", "years", "wrong", "wrap", "white", "west", "week", "vlog", "view", 
    "version", "vault", "usa", "uma", "ultimate", "tutorial", "triple", "thomas", "team", "tap", 
    "style", "stroke", "spin", "south", "song", "skills", "shot", "setting", "set", "session", "service", "series",
    "seconds", "san", "sam", "ryan", "roll", "review", "real", "range", "race", "properly", "proper", "project",
    "professional","process", "pro", "power", "pong", "point", "played", "play", "ping", "pit", "pick", 
    "performance", "perfect", "pedido", "paul", "pass", "parte", "para", "open", "national", "mini",
    "michael", "max", "matt", "master", "mark", "long", "lol", "live", "lesson", "learning", "learn", 
    "las", "lapse", "justin", "june", "july", "john", "joe", "jerk", "james", "jack", "jacks", "install",
    "hot", "hd", "guide", "great", "good", "gangnam", "funny", "free", "form", "footage", "flip", "flat",
    "final", "fidget", "fazer", "fail", "extreme", "episode", "epic", "easy", "drop", "diy", "demo",
    "demonstration", "del", "decorating", "david", "curl", "crazy", "cover", "control", "contact",
    "compilation", "comment", "collection", "classic", "civil", "chris", "chip", "charleston", "change",
    "championship", "championships", "challenge", "care", "break", "big", "ben", "beginners", "beginner",
    "base", "basic", "basics", "bad", "awesome", "asmr", "april", "apply", "amazing", "alex", "action",
    "academy", "à", "||", "~", "playing", "cheating", "fling", "qualifying", "attending", "attempting",
    "arlington", "ping", "pong", "nottingham", "lexington", "year", "ying", "air", "adding", "adjusting",
    "accepting", "alternating", "amazingly", "applying", "arranging", "ascending", "backing", "banding", "banging"
    "banking", "beijing", "birmingham", "bing", "boeing", "burlington", "changing", "center", "centering",
    "tai", "chi", "como", "coming", "choose", "choosing", "hd", "jack", "jacks", "jackson", "jordan", "box", "great", "video", "videos",
    "wilmington", "farmington", "remington", "sexy", "jason", "ingrid", "jing", "jan", "wyoming", "starring", "singh", "sarah", "nanjing", 
    "ming", "mary"
    ]


    add = ["pole vault", "ping pong", "jumping jack", "gangnam style", "dropping", "rubiks cube", "tai chi", "shot put"]

    # create_visual_vocab()
    # probe_visual_vocab()

    # remove_folders(remove)
    # save_imgs(add)

    # print(len(glob.glob(f'{visual_vocab_dir}/*')))

    # get_visual_vocab_paths()

    # probe_visual_vocab()

    # get_visual_vocab_tag_positives(add, remove)

    tags_dict = pickle.load(open('tag_to_imgs.pickle', "rb"))
    imgs_dict = pickle.load(open('img_to_tags.pickle', "rb"))

    # print(list(tags_dict.keys()))
    # print(tags_dict['falling'])
    # print(len(tags_dict))
    # print(len(imgs_dict))
    # print(imgs_dict['5qjUifx4FDA'])

    # min_vals = float('inf')
    # for k in tags_dict:
    #     min_vals = min(min_vals, len(tags_dict[k]))
    # print(min_vals)

    probe_tags_dict("relieving")