?($	2J?Ly?????x??^???sF????!6?>W[???$	?ٵ?t|@????:?@e}՝q?@!f???P0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?sF??????ڊ?e??AHP?s???Y??(???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????????镲??A??^)??Y?N@aã?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb????K7?A??A/?$???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?>W[????k	??g??A??Q???YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#J{?/L????:M??Aۊ?e????Y<?R?!???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????'1?Z??A46<???Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?$??C??????o??A?J?4??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??g??s?????N@??Aq?-???Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?p=
ף??????????A?!??u???Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??C?l????S㥛???A[????<??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
e?`TR'???f??j+??A?HP???Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????L?
F%u??Ay?&1???Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c?????h o???A	?c?Z??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f??a????9??m4???A?MbX9??Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ?????u????A"lxz?,??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~???????B??A?n?????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?d?`TR??yX?5?;??Affffff??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM????Ǻ????A?JY?8???Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o?ŏ1??e?`TR'??AQ?|a2??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??????ݵ?|г??A??k	????Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#J{?/L?????1????A+?????Y?c?]Kȷ?*	gffffR?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\ A?c???!B?F?<@)??#?????1{?p???:@:Preprocessing2F
Iterator::Model3ı.n???!į?v%lA@)	?c???1?????5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???~?:??!??D?IP@)??V?/???1??\??3@:Preprocessing2U
Iterator::Model::ParallelMapV2c?=yX??!?k??^R*@)c?=yX??1?k??^R*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???N@??!?}$@)???N@??1?}$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate+?پ?!??"r?(@)??q????1R!Gȿ-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??j+????!???~?b1@)=?U?????1a?U???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ׁsF??!.n?<)S @)??ׁsF??1.n?<)S @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9-XC??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?Zr????????????ڊ?e??!?k	??g??	!       "	!       *	!       2$	?A??????'Q?Т˴??n?????!ۊ?e????:	!       B	!       J$	"PdU???<q?@4???W[?????!?c?]Kȷ?R	!       Z$	"PdU???<q?@4???W[?????!?c?]Kȷ?JCPU_ONLYY-XC??@b Y      Y@q?ۛ4?U@"?
both?Your program is POTENTIALLY input-bound because 52.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.0244% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 