?($	?7????Ѧ?L?2???c?ZB??!????_v??$	????@??U?I@?@??@!?q0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?c?ZB???Q?????A=?U????YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@a??+??^?I+??A5?8EGr??Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&pΈ??????S㥛???A?????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	???EGr????Ash??|???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??j+??????x?&1??ATR'?????Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=
ףp=????/?$??AZd;?O??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??1??%??c?ZB>???AF%u???Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????_v??M?O???AmV}??b??Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2??%?????-????A)?Ǻ???Y??g??s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?(??0????/?$??A?6?[ ??Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?3??7??NbX9???A}??b???Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM???ZӼ???AJ{?/L???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ??!?lV}??A?(\?????Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?鷯??`vOj??A???_vO??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L?
F%u??????o??A?c]?F??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??s???|??Pk???AԚ?????Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F????????A?I+???Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??s????R?!?u??A[B>?٬??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?(??0??aTR'????A??#?????YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q?|??.?!??u??A A?c?]??Y??g??s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n????Q?|??Ac?ZB>???Y?g??s???*	43333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?A`??"??!?$n??]A@)?C??????1??q?m@@:Preprocessing2F
Iterator::Model9??m4???!??^'T?A@)      ??1?4Nz??5@:Preprocessing2U
Iterator::Model::ParallelMapV2?X?? ??!aߨ1:+@)?X?? ??1aߨ1:+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\ A?c???!??P??&P@)??ͪ?ն?1? ????$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?St$????!<?F???.@)???????1?8?I?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??z6???!?????@)??z6???1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/?$???!??????3@)/n????1? ???Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??_vO??!? ſo@)??_vO??1? ſo@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no99? ?6=@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?1C????????	?t???Q?????!?ZӼ???	!       "	!       *	!       2$	h∞@C????ѥtd??=?U????![B>?٬??:	!       B	!       J$	{
%???_fp??????Mb??!??+e???R	!       Z$	{
%???_fp??????Mb??!??+e???JCPU_ONLYY9? ?6=@b Y      Y@q???4)?>@"?
both?Your program is POTENTIALLY input-bound because 45.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.7467% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 