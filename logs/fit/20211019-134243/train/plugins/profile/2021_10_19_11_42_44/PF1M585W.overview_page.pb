?($		i?^?	@?&??9? @?n?????!V????OC@$	?E@??@?@??^@?u??N??!5pr?f?(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$3ı.n????W[?????A?W[?????Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V????OC@?=?U/A@AB?f???@Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??S㥛??V????_??A?:pΈ???Y?,C????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(?#@?9#J{?"@A??ǘ????YI.?!????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Q?|a2????d?`T??A??	h"??Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|гY???A`??"??A46<???YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?h o???䃞ͪ???AȘ?????Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}????X?5?;N??A?T???N??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=?U??????S????A?O??e??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	M??St$???镲q??A???QI???Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
&S??:??f??a????A??H.?!??Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?J???[B>?٬??AGx$(??YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q?|???W[?????A?O??n??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???H???H?}8??A??<,Ԛ??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ??????ŏ1w??A?lV}???Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?<,Ԛ????c?ZB??AO??e?c??Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ??8gDio??AQ?|a??Y`??"????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&\ A?c????? ?rh??AT㥛? ??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_)?Ǻ??z6?>W??AR???Q??Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?J????ǘ?????A?:pΈ??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?n??????e??a???A????K7??Y?+e?X??*	????̎?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?QI??&??!E???V.B@)?T???N??1??R}G?@@:Preprocessing2F
Iterator::Model??(\????!m???A@)?q??????1???[66@:Preprocessing2U
Iterator::Model::ParallelMapV2?y?):???!?< ?6+@)?y?):???1?< ?6+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????!?????P@)+?پ?1i???nr%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??1??%??!?yy?@)??1??%??1?yy?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateгY?????!?U\ϊ7*@) o?ŏ??1???%Yk@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ǘ????!?Ž>f1@)Ǻ?????1??|??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??H?}??!????@)??H?}??1????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 80.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???y?(??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	3?X?/?@?5y??%@?e??a???!?=?U/A@	!       "	!       *	!       2$	?4j?????{Z%??????K7??!B?f???@:	!       B	!       J$	?vtI«???B?????+e?X??!?? ?rh??R	!       Z$	?vtI«???B?????+e?X??!?? ?rh??JCPU_ONLYY???y?(??b Y      Y@q$?m0T@"?
both?Your program is POTENTIALLY input-bound because 80.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?80.4873% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 