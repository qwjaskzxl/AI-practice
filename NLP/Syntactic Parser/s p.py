import os
import pandas as pd
import glob
import json
import logging
import socket
import subprocess
import sys
import time
import psutil
import requests
import nltk
import tempfile
import re
from subprocess import PIPE
from six import text_type
from nltk.internals import find_jar_iter, config_java, java, _java_options, find_jars_within_path
from nltk.parse.api import ParserI

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
_stanford_url = 'https://nlp.stanford.edu/software/lex-parser.shtml'
nltk.internals.config_java("D:/Program Files/Java/jdk1.8.0_111/bin/java.exe")
java_path = "D:/Program Files/Java/jdk1.8.0_111/bin/java.exe"
os.environ['JAVAHOME'] = java_path


class GenericStanfordParser(ParserI):
    """Interface to the Stanford Parser"""

    _MODEL_JAR_PATTERN = r'stanford-parser-(\d+)(\.(\d+))+-models\.jar'
    _JAR = r'stanford-parser\.jar'
    _MAIN_CLASS = 'edu.stanford.nlp.parser.lexparser.LexicalizedParser'

    _USE_STDIN = False
    _DOUBLE_SPACED_OUTPUT = False

    def __init__(self, path_to_jar=None, path_to_models_jar=None,
                 model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
                 encoding='utf8', verbose=False,
                 java_options='-mx1000m', corenlp_options=''):

        # find the most recent code and model jar
        stanford_jar = max(
            find_jar_iter(
                self._JAR, path_to_jar,
                env_vars=('STANFORD_PARSER', 'STANFORD_CORENLP'),
                searchpath=(), url=_stanford_url,
                verbose=verbose, is_regex=True
            ),
            key=lambda model_path: os.path.dirname(model_path)
        )

        model_jar = max(
            find_jar_iter(
                self._MODEL_JAR_PATTERN, path_to_models_jar,
                env_vars=('STANFORD_MODELS', 'STANFORD_CORENLP'),
                searchpath=(), url=_stanford_url,
                verbose=verbose, is_regex=True
            ),
            key=lambda model_path: os.path.dirname(model_path)
        )

        # Adding logging jar files to classpath
        stanford_dir = os.path.split(stanford_jar)[0]
        self._classpath = tuple([model_jar] + find_jars_within_path(stanford_dir))

        self.model_path = model_path
        self._encoding = encoding
        self.corenlp_options = corenlp_options
        self.java_options = java_options

    def _parse_trees_output(self, output_):
        res = []
        cur_lines = []
        cur_trees = []
        blank = False
        for line in output_.splitlines(False):
            if line == '':
                if blank:
                    res.append(iter(cur_trees))
                    cur_trees = []
                    blank = False
                elif self._DOUBLE_SPACED_OUTPUT:
                    cur_trees.append(self._make_tree('\n'.join(cur_lines)))
                    cur_lines = []
                    blank = True
                else:
                    res.append(iter([self._make_tree('\n'.join(cur_lines))]))
                    cur_lines = []
            else:
                cur_lines.append(line)
                blank = False
        return iter(res)

    def parse_sents(self, sentences, verbose=False):

        cmd = [
            self._MAIN_CLASS,
            '-model', self.model_path,
            '-sentences', 'newline',
            '-outputFormat', self._OUTPUT_FORMAT,
            '-tokenized',
            '-escaper', 'edu.stanford.nlp.process.PTBEscapingProcessor',
        ]
        return self._parse_trees_output(self._execute(
            cmd, '\n'.join(' '.join(sentence) for sentence in sentences), verbose))

    def raw_parse(self, sentence, verbose=False):

        return next(self.raw_parse_sents([sentence], verbose))

    def raw_parse_sents(self, sentences, verbose=False):

        cmd = [
            self._MAIN_CLASS,
            '-model', self.model_path,
            '-sentences', 'newline',
            '-outputFormat', self._OUTPUT_FORMAT,
        ]
        return self._parse_trees_output(self._execute(cmd, '\n'.join(sentences), verbose))

    def tagged_parse(self, sentence, verbose=False):

        return next(self.tagged_parse_sents([sentence], verbose))

    def tagged_parse_sents(self, sentences, verbose=False):

        tag_separator = '/'
        cmd = [
            self._MAIN_CLASS,
            '-model', self.model_path,
            '-sentences', 'newline',
            '-outputFormat', self._OUTPUT_FORMAT,
            '-tokenized',
            '-tagSeparator', tag_separator,
            '-tokenizerFactory', 'edu.stanford.nlp.process.WhitespaceTokenizer',
            '-tokenizerMethod', 'newCoreLabelTokenizerFactory',
        ]
        # We don't need to escape slashes as "splitting is done on the last instance of the character in the token"
        return self._parse_trees_output(self._execute(
            cmd, '\n'.join(' '.join(tag_separator.join(tagged) for tagged in sentence) for sentence in sentences), verbose))

    def _execute(self, cmd, input_, verbose=False):
        encoding = self._encoding
        cmd.extend(['-encoding', encoding])
        if self.corenlp_options:
            cmd.append(self.corenlp_options)

        default_options = ' '.join(_java_options)

        # Configure java.
        config_java(options=self.java_options, verbose=verbose)

        # Windows is incompatible with NamedTemporaryFile() without passing in delete=False.
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as input_file:
            # Write the actual sentences to the temporary input file
            if isinstance(input_, text_type) and encoding:
                input_ = input_.encode(encoding)
            input_file.write(input_)
            input_file.flush()

            # Run the tagger and get the output.
            if self._USE_STDIN:
                input_file.seek(0)
                stdout, stderr = java(cmd, classpath=self._classpath,
                                      stdin=input_file, stdout=PIPE, stderr=PIPE)
            else:
                cmd.append(input_file.name)
                stdout, stderr = java(cmd, classpath=self._classpath,
                                      stdout=PIPE, stderr=PIPE)

            stdout = stdout.replace(b'\xc2\xa0', b' ')
            stdout = stdout.replace(b'\x00\xa0', b' ')
            stdout = stdout.decode(encoding)

        os.unlink(input_file.name)

        # Return java configurations to their default values.
        config_java(options=default_options, verbose=False)

        return stdout


class StanfordParser(GenericStanfordParser):
    _OUTPUT_FORMAT = 'penn'

    def _make_tree(self, result):
        return Tree.fromstring(result)


class StanfordCoreNLP:
    def __init__(self, path_or_host, port=None, memory='4g', lang='en', timeout=1500, quiet=True,
                 logging_level=logging.WARNING):
        self.path_or_host = path_or_host
        self.port = port
        self.memory = memory
        self.lang = lang
        self.timeout = timeout
        self.quiet = quiet
        self.logging_level = logging_level

        logging.basicConfig(level=self.logging_level)

        # Check args
        self._check_args()

        if path_or_host.startswith('http'):
            self.url = path_or_host + ':' + str(port)
            logging.info('Using an existing server {}'.format(self.url))
        else:

            # Check Java
            if not subprocess.call(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) == 0:
                raise RuntimeError('Java not found.')

            # Check if the dir exists
            if not os.path.isdir(self.path_or_host):
                raise IOError(str(self.path_or_host) + ' is not a directory.')
            directory = os.path.normpath(self.path_or_host) + os.sep
            self.class_path_dir = directory

            # Check if the language specific model file exists
            switcher = {
                'en': 'stanford-corenlp-[0-9].[0-9].[0-9]-models.jar',
                'zh': 'stanford-chinese-corenlp-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-models.jar',
                'ar': 'stanford-arabic-corenlp-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-models.jar',
                'fr': 'stanford-french-corenlp-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-models.jar',
                'de': 'stanford-german-corenlp-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-models.jar',
                'es': 'stanford-spanish-corenlp-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-models.jar'
            }
            jars = {
                'en': 'stanford-corenlp-x.x.x-models.jar',
                'zh': 'stanford-chinese-corenlp-yyyy-MM-dd-models.jar',
                'ar': 'stanford-arabic-corenlp-yyyy-MM-dd-models.jar',
                'fr': 'stanford-french-corenlp-yyyy-MM-dd-models.jar',
                'de': 'stanford-german-corenlp-yyyy-MM-dd-models.jar',
                'es': 'stanford-spanish-corenlp-yyyy-MM-dd-models.jar'
            }
            if len(glob.glob(directory + switcher.get(self.lang))) <= 0:
                raise IOError(jars.get(
                    self.lang) + ' not exists. You should download and place it in the ' + directory + ' first.')

            # If port not set, auto select
            if self.port is None:
                for port_candidate in range(9000, 65535):
                    if port_candidate not in [conn.laddr[1] for conn in psutil.net_connections()]:
                        self.port = port_candidate
                        break

            # Check if the port is in use
            if self.port in [conn.laddr[1] for conn in psutil.net_connections()]:
                raise IOError('Port ' + str(self.port) + ' is already in use.')

            # Start native server
            logging.info('Initializing native server...')
            cmd = "java"
            java_args = "-Xmx{}".format(self.memory)
            java_class = "edu.stanford.nlp.pipeline.StanfordCoreNLPServer"
            class_path = '"{}*"'.format(directory)

            args = [cmd, java_args, '-cp', class_path, java_class, '-port', str(self.port)]

            args = ' '.join(args)

            logging.info(args)

            # Silence
            with open(os.devnull, 'w') as null_file:
                out_file = None
                if self.quiet:
                    out_file = null_file

                self.p = subprocess.Popen(args, shell=True, stdout=out_file, stderr=subprocess.STDOUT)
                logging.info('Server shell PID: {}'.format(self.p.pid))

            self.url = 'http://localhost:' + str(self.port)

        # Wait until server starts
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_name = urlparse(self.url).hostname
        time.sleep(1)  # OSX, not tested
        while sock.connect_ex((host_name, self.port)):
            logging.info('Waiting until the server is available.')
            time.sleep(1)
        logging.info('The server is available.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        logging.info('Cleanup...')
        if hasattr(self, 'p'):
            try:
                parent = psutil.Process(self.p.pid)
            except psutil.NoSuchProcess:
                logging.info('No process: {}'.format(self.p.pid))
                return

            if self.class_path_dir not in ' '.join(parent.cmdline()):
                logging.info('Process not in: {}'.format(parent.cmdline()))
                return

            children = parent.children(recursive=True)
            for process in children:
                logging.info('Killing pid: {}, cmdline: {}'.format(process.pid, process.cmdline()))
                # process.send_signal(signal.SIGTERM)
                process.kill()

            logging.info('Killing shell pid: {}, cmdline: {}'.format(parent.pid, parent.cmdline()))
            # parent.send_signal(signal.SIGTERM)
            parent.kill()

    def annotate(self, text, properties=None):
        if sys.version_info.major >= 3:
            text = text.encode('utf-8')

        r = requests.post(self.url, params={'properties': str(properties)}, data=text,
                          headers={'Connection': 'close'})
        return r.text

    def tregex(self, sentence, pattern):
        tregex_url = self.url + '/tregex'
        r_dict = self._request(tregex_url, pattern, "tokenize,ssplit,depparse,parse", sentence)
        return r_dict

    def tokensregex(self, sentence, pattern):
        tokensregex_url = self.url + '/tokensregex'
        r_dict = self._request(tokensregex_url, pattern, "tokenize,ssplit,depparse", sentence)
        return r_dict

    def semgrex(self, sentence, pattern):
        semgrex_url = self.url + '/semgrex'
        r_dict = self._request(semgrex_url, pattern, "tokenize,ssplit,depparse", sentence)
        return r_dict

    def word_tokenize(self, sentence, span=False):
        r_dict = self._request('ssplit,tokenize', sentence)
        tokens = [token['originalText'] for s in r_dict['sentences'] for token in s['tokens']]

        # Whether return token span
        if span:
            spans = [(token['characterOffsetBegin'], token['characterOffsetEnd']) for s in r_dict['sentences'] for token
                     in s['tokens']]
            return tokens, spans
        else:
            return tokens

    def pos_tag(self, sentence):
        r_dict = self._request('pos', sentence)
        words = []
        tags = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['originalText'])
                tags.append(token['pos'])
        return list(zip(words, tags))

    def ner(self, sentence):
        r_dict = self._request('ner', sentence)
        words = []
        ner_tags = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['originalText'])
                ner_tags.append(token['ner'])
        return list(zip(words, ner_tags))

    def parse(self, sentence):
        r_dict = self._request('pos,parse', sentence)
        return [s['parse'] for s in r_dict['sentences']][0]

    def dependency_parse(self, sentence):
        r_dict = self._request('depparse', sentence)
        return [(dep['dep'], dep['governor'], dep['dependent']) for s in r_dict['sentences'] for dep in
                s['basicDependencies']]

    def coref(self, text):
        r_dict = self._request('coref', text)

        corefs = []
        for k, mentions in r_dict['corefs'].items():
            simplified_mentions = []
            for m in mentions:
                simplified_mentions.append((m['sentNum'], m['startIndex'], m['endIndex'], m['text']))
            corefs.append(simplified_mentions)
        return corefs

    def switch_language(self, language="en"):
        self._check_language(language)
        self.lang = language

    def _request(self, annotators=None, data=None, *args, **kwargs):
        if sys.version_info.major >= 3:
            data = data.encode('utf-8')

        properties = {'annotators': annotators, 'outputFormat': 'json'}
        params = {'properties': str(properties), 'pipelineLanguage': self.lang}
        if 'pattern' in kwargs:
            params = {"pattern": kwargs['pattern'], 'properties': str(properties), 'pipelineLanguage': self.lang}

        logging.info(params)
        r = requests.post(self.url, params=params, data=data, headers={'Connection': 'close'})
        r_dict = json.loads(r.text)

        return r_dict

    def _check_args(self):
        self._check_language(self.lang)
        if not re.match('\dg', self.memory):
            raise ValueError('memory=' + self.memory + ' not supported. Use 4g, 6g, 8g and etc. ')

    def _check_language(self, lang):
        if lang not in ['en', 'zh', 'ar', 'fr', 'de', 'es']:
            raise ValueError('lang=' + self.lang + ' not supported. Use English(en), Chinese(zh), Arabic(ar), '
                                                   'French(fr), German(de), Spanish(es).')

eng_parser = StanfordParser("stanford-parser.jar",
                            "stanford-parser-3.9.1-models.jar",
                            "englishPCFG.ser.gz")

df = pd.read_excel('test.xlsx')
p = {'parse': []}
for s in df['E'][:1]:
    #     print(s)
    #     print(str(list(eng_parser.parse(s.split()))))
    try:
        p['parse'].append(str(list(eng_parser.parse(s.split()))))
    except:
        p['parse'].append('')
    pd.DataFrame(p).to_excel('parse_.xlsx')

# visualize
from nltk.tree import Tree

nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')
s = 'At the end of the day, successfully launching a new product means reaching the right audience and consistently delivering a very convincing message. To avoid spending money recklessly because of disjointed strategies, we have developed several recommendations.'
# print ('Tokenize:', nlp.word_tokenize(s))
# print ('Part of Speech:', nlp.pos_tag(s))
# print ('Named Entities:', nlp.ner(s))
print('Constituency Parsing:', nlp.parse(s))
# print ('Dependency Parsing:', nlp.dependency_parse(s))
tree = Tree.fromstring(nlp.parse(s))
tree.draw()

# words = s.split(' ')
layer1 = []
for n in tree[0]:
    n = str(n)
    sub = ''
    for w in n.split():
        if w.strip(')') in s:
            sub += w.strip(')')+' '
    layer1.append(sub.strip())
print()

nlp.close()