?($	r???"???ɐ?????L?
F%u??!??? ?r??$	?OH?@\?ﻤ?@Y?y??1@!??݉V,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$2U0*???b??4?8??A???Q???Y???9#J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u????Zd;???Az?,C???YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ???ۊ?e????A?#??????Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&y?&1???Gr?????A?R?!?u??Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ???镲q??A??\m????YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F??+??ݓ???AV}??b??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_vO??46<???A??^??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&KY?8?????? ?	??A?q??????Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n???w-!?l??A?*??	??Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??? ?r??x??#????A9??m4???Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?-???????N@??A??S㥛??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{???C??6??A???9#J??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	?c??'1?Z??A??(???Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ё\?C???????Mb??A??_?L??Y?V-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??0?*??F%u???A???1????Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&TR'?????M?O???A?>W[????Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ꕲq???u????A?:pΈ??Y o?ŏ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O??L7?A`???A?z6?>??Y?j+??ݣ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&tF??_???Q???A?&?W??Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&? ?rh???(~??k	??A???????Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L?
F%u??j?q?????A??????Yu????*	     :?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??y?)??!?(??(?B@)vq?-??1I???"A@:Preprocessing2F
Iterator::Model??a??4??!?q?q@@)??~j?t??1?P&?t3@:Preprocessing2U
Iterator::Model::ParallelMapV2?D?????!??&%??*@)?D?????1??&%??*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????B???!r?q?P@)??HP??1AA$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateM?O???!Tg?x?P-@)Q?|a2??1??S?r
@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????<,??!%7)??@)????<,??1%7)??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?MbX9??!      4@)Gx$(??1W1??^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*X9??v???!??{?~@)X9??v???1??{?~@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9A?^H,?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???S????ŉMW??j?q?????!x??#????	!       "	!       *	!       2$	o>?5?*????T????????!??(???:	!       B	!       J$	+Sx;???WN????|???@??ǘ?!???9#J??R	!       Z$	+Sx;???WN????|???@??ǘ?!???9#J??JCPU_ONLYYA?^H,?@b Y      Y@q?5???S@"?
both?Your program is POTENTIALLY input-bound because 55.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?79.2488% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 