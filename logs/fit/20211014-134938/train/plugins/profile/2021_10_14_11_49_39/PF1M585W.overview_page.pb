?($	1=3dd??В????lxz?,C??!???????$	ק?X??@?6???g@Te??l??!?s???*@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$e?X???,e?X??AȘ?????Y?????K??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&TR'???????ͪ????A?H.?!???Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	???U0*????A?????M??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[??V-?????AP??n???Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<??????????AjM????YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&333333???Zd;???Ax$(~??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Mb???h o???A??H.???Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?c???Pk?w???A?S㥛???Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?K7?A`???L?J???A?镲q??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	{?/L?
??2U0*???A??&S??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?#??????Nё\?C??A?St$????YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K???(~??k	??A?(\?????YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&      ??f??a????A'1?Z??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H?z?G??f?c]?F??A?ZB>????Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF?????&??A?? ?	??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????JY?8???A????߾??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e?c]??8??d?`??AO??e?c??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%u???F%u???A??V?/???YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ??K?46??A>?٬?\??YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C???????? ?rh??AL7?A`???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&lxz?,C??pΈ?????A??H.???YQ?|a2??*	fffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?#??????!1???]B@)???B?i??1e3m??@@:Preprocessing2F
Iterator::Model??????!?6??%?@@)?f??j+??1L?5M?g4@:Preprocessing2U
Iterator::Model::ParallelMapV2?x?&1??!~>???)@)?x?&1??1~>???)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?H?}??!?d3m?P@)9EGr???1??jN_'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?;Nё\??!8ޛr?@)?;Nё\??18ޛr?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?!?uq??!???xo?*@)_?Qګ?1??'ݢ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM?J???!Q?d3m2@)??ܥ?1M???]@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*=?U?????!?ܩ@??@)=?U?????1?ܩ@??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9˷QJ&?
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	b?F?????.?v/???,e?X??!?JY?8???	!       "	!       *	!       2$	]El??Q??~?M??????H.???!??&S??:	!       B	!       J$	z6?>W??e?V?+z?????H??!?????K??R	!       Z$	z6?>W??e?V?+z?????H??!?????K??JCPU_ONLYY˷QJ&?
@b Y      Y@q-N?0C@"?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.375% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 