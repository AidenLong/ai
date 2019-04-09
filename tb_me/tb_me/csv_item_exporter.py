# -*- coding:utf-8 -*-

# 调用scrapy中的setting
from scrapy.conf import settings
# 调用CsvItemExporter
from scrapy.contrib.exporter import CsvItemExporter


class TbCsvItemExpoter(CsvItemExporter):
    def __init__(self, *args, **kwargs):
        delimiter = settings.get("CSV_DELIMITER")
        kwargs['delimiter'] = delimiter
        fields_to_export = settings.get('FIELDS_TO_EXPORT', [])
        if fields_to_export:
            kwargs['fields_to_export'] = fields_to_export
        super(TbCsvItemExpoter, self).__init__(*args, **kwargs)
