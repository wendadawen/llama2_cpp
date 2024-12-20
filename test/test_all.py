"""
Run simply with
$ pytest
"""
import os
import subprocess

expected_stdout = b'Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily\'s mom said, "Lily, let\'s go to the park." Lily was sad and didn\'t know what to do. She said, "I want to play with your ball, but I can\'t find it."\nLily was sad and didn\'t know what to do. She said, "I\'m sorry, Lily. I didn\'t know what to do."\nLily didn\'t want to help her mom, so she'

# -----------------------------------------------------------------------------
# actual tests

def test_main():
    """ Forwards a model against a known-good desired outcome in run.c for 200 steps"""

    model_path = os.path.join("stories260K.bin")
    tokenizer_path = os.path.join("tok512.bin")
    command = ["./go.exe", model_path, "-z", tokenizer_path, "-t", "0.0", "-n", "200"]
    with open('./test/stderr.txt', mode='wb') as fe:
        with open('./test/stdout.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)  #pipe in windows terminal does funny things like replacing \n with \r\n
            proc.wait()

    with open('stdout.txt', mode='r') as f:
        stdout = f.read()
    # strip the very last \n that is added by run.c for aesthetic reasons
    stdout = stdout[:-1].encode('ascii')

    assert stdout == expected_stdout
