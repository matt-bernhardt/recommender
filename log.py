# -*- coding: utf-8 -*-
from __future__ import absolute_import


class Log():

    def __init__(self, name):
        self.name = name
        self.file = open(name, 'w')
        return None

    def end(self):
        self.file.close()
        self.file = None
        return self.file

    def message(self, msg):
        return self.file.write(msg + '\n')

    def summarizeData(self, data):
        self.message('data type: ' + str(type(data)))
        self.message('data type: ' + str(type(data)))
        self.message('rows:      ' + str(len(data)))
        self.message('fields:    ' + str(len(data[0])))
        self.message('feild name:')
        self.message(str(data[0]))
        self.message('sample record:')
        self.message(str(data[1]))

    def summarizeModel(self, data):
        self.message('model type:     ' + str(type(data)))
        self.message('model topics:   ' + str(len(data.components_)))
        self.message('model features: ' + str(len(data.components_[0])))
        self.message('sample topic:')
        self.message(str(data.components_[0]))
        self.message('model parameters:')
        self.message(str(data.get_params()))
        self.message('')

    def summarizeTF(self, data):
        self.message('matrix type:      ' + str(type(data)))
        self.message('tf record count:  ' + str(len(data.toarray())))
        self.message('tf record length: ' + str(len(data.toarray()[0])) + ' features')
        self.message('sample record:    ' + str(data.toarray()[0]))
        self.message('matrix excerpt:\n' + str(data.toarray()))
        self.message('')

    def summarizeUniverse(self, data):
        self.message('data type: ' + str(type(data)))
        self.message('data points: ' + str(len(data)))
        self.message('dimensions:  ' + str(len(data[0])))
        self.message('sample data: ' + str(data[0]))
        self.message('excerpt:')
        self.message(str(data))
        self.message('')
