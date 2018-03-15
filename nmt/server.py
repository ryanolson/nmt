# -*- coding: utf-8 -*-
import glob
import logging
import os
import random
import threading
import time

import grpc
from concurrent import futures


import sys
import time
import functools
#from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool

#import numpy as np
#import scipy.io.wavfile

import translation_pb2
import translation_pb2_grpc


def get_logger(name, level=None):
    level = level or logging.INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

log = get_logger("translation_grpc", level=logging.DEBUG)
MAX_GRPC_WORKERS = int(os.environ.get("GRPC_MAX_WORKERS", 1))


class InferenceServer(translation_pb2_grpc.InferenceServicer):

    def __init__(self, compute_fn):
        log.info("initialzing translation grpc infernece service")
        self.compute_fn = compute_fn

    def Compute(self, request, context):
        log.info("inference request: {}".format(str(request.batch_id)))
        translations = self.compute_fn(request)
        log.info("inference finished")
        return self.Results(request.input_sentences, translations, batch_id=request.batch_id)

    def Results(self, inputs, translations, batch_id=None):
        predictions = translation_pb2.BatchPredictions(batch_id=batch_id)
        for input, output in zip(inputs, translations):
            t = predictions.translations.add()
            t.input = input
            t.output = output
        #for words, times in decoded:
        #    transcription = predictions.audio_transcriptions.add()
        #    for word, time in zip(words, times):
        #        transcription.utterances.add(word=word, offset=time)
        log.info("inference results:\n{}".format(str(predictions)))
        return predictions

def run(engine=None):
    engine = engine or InferenceServer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_GRPC_WORKERS))
    translation_pb2_grpc.add_InferenceServicer_to_server(engine, server)
    server.add_insecure_port('[::]:50051')
    log.info("starting service")
    server.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
