?($	QdU??????*6u???~8gDi??!?3??7??$	MΠ?9?@??@YJ?@o[?Re?@!Sbo?t?/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$F??_????lV}???A?;Nё\??Y?m4??@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)\???(???3??7??AB`??"???Y'?Wʢ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????O??n??A&䃞ͪ??Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c???w??/???A?ŏ1w??Y??|гY??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?2ı.n??M?O???Ah??s???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m??`vOj??At$???~??Y?A`??"??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&333333??[Ӽ???A??{??P??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ݓ??Z??F????x??A??ڊ?e??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG??,Ԛ????A/?$???Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??ܵ??????߾??A?Zd;??Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
9??v??????^)??A?c?]K???Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7????	h"l??A`vOj??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n?????'????A<Nё\???Y??H.?!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&㥛? ????H.?!???A?????YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Q??O@a????A/?$????Y ?o_Ω?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?x?&1??vq?-??A?(??0??Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8EGr???f?c]?F??A??St$???Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d?]K??????????A46<?R??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??g????S?!?uq??AQ?|a2??Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m???9#J{???A$???????Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~8gDi??tF??_??A????z??YA??ǘ???*	gffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?c]?F??!???A?A@)?X?? ??1ڡ??"@@:Preprocessing2F
Iterator::Model???????!??BR?C@)M?O????1??zFߴ9@:Preprocessing2U
Iterator::Model::ParallelMapV2??|?5^??!w???*@)??|?5^??1w???*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipU0*????!A)???nN@)?b?=y??1??r?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice/n????!{?	-,Z@)/n????1{?	-,Z@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatevq?-??!??9?'@)y?&1???1???F?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?H.?!???!??-???0@)?]K?=??1T& ???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*2U0*???!??r??@)2U0*???1??r??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	T?fj?????y?[??tF??_??!??	h"l??	!       "	!       *	!       2$	?!?o?n???F?????????z??!`vOj??:	!       B	!       J$	?D?RW'??SvU?t???e??a???!?m4??@??R	!       Z$	?D?RW'??SvU?t???e??a???!?m4??@??JCPU_ONLYY??????@b Y      Y@q\?hYn8;@"?
both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?27.2204% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 