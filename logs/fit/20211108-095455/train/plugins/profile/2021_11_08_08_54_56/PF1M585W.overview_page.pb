?($	???ڕY??〿;?&??X9??v??!
ףp=? @$	d??^?@u????@]/?Y??@!j??F2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??	h"????W?2ı?AǺ????Yq=
ףp??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?߾?3??c?ZB>???AZd;?O??Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}гY????OjM???A:#J{?/??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?L?J????????A}?5^?I??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C??????$(~??k??A[Ӽ???Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=? @??HP??A<Nё\???Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F???4?8EG??A?&?W??Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?镲q???W?2ı??A??Pk?w??Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)\???(??ꕲq???AM??St$??Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?=yX?5??????H??A䃞ͪ???Y????镢?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?e??a??????h o??A)?Ǻ???Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"??u????M?O???A>yX?5???Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q?-????3??7??A?D?????Y?Zd;??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?鷯??sh??|???A?ǘ?????Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&b??4?8??'1?Z??A]?C?????Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????K7?????ׁs??A=
ףp=??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~8gDi??+??ݓ???A^?I+??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;?????V?/???As??A???Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&KY?8??????(????A??W?2???Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?+e?X????ʡE??A?&?W??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v??M?J???Aq???h ??YK?=?U??*	    l?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ڊ?e??!U???IEA@)?s????1E???T??@:Preprocessing2F
Iterator::Model?ׁsF???!???S??@@)d?]K???1??c?@?2@:Preprocessing2U
Iterator::Model::ParallelMapV2T㥛? ??!|˷|˷,@)T㥛? ??1|˷|˷,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'?W???!?V??P@)c?ZB>???1q?a?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??1??%??!???T??!@)??1??%??1???T??!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate}гY????!??V ?0@)X?5?;N??1
?%???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap~8gDi??!~?????4@)46<???1?
??
?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?z6?>??!K???@)?z6?>??1K???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?C^?1y@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???Wqi??ߜ?B~|??M?J???!??HP??	!       "	!       *	!       2$	۠? W???????%??q???h ??!<Nё\???:	!       B	!       J$	???3??{??B???W[?????!q=
ףp??R	!       Z$	???3??{??B???W[?????!q=
ףp??JCPU_ONLYY?C^?1y@b Y      Y@q???W??>@"?
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.914% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 