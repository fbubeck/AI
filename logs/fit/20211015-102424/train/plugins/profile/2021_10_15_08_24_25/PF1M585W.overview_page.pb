?($	{i`}????????9?????ͪ????!???T????$	??Á?@?w?JR@? ڡl???!?>S=A?+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$8gDio???HP???AV-????YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M???0?*??A$(~????Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6??Q?|a2??Ak?w??#??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????e?`TR'??A?s????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U??'1?Z??AZd;?O??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(??????Q???AZd;?O???Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??+e???Q?|a2??A-??????Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c?????k	????A??|гY??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'?W????*??	??A???߾??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?d?`TR????/?$??As??A???YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?^)????G?z???Aŏ1w-??Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Q?|a???QI??&??A?R?!?u??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S???M?O???Aa??+e??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5?????HP??A??_vO??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?o_???M?J???A?=yX?5??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?X???aTR'????Aё\?C???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@?߾???R???Q??A?'????Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T???????????A=,Ԛ???Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??N@a?????Mb??A}?5^?I??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e?c?????ZӼ??A+??ݓ???Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ͪ???????x?&??A?A?f???Y?/?'??*	gffffr?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ǘ????! #6??
B@)?t?V??1C?*N@@:Preprocessing2F
Iterator::Model?!??u???!??.BQh?@)X?5?;N??1:??|?2@:Preprocessing2U
Iterator::Model::ParallelMapV2M??St$??!͒???+)@)M??St$??1͒???+)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipO??e???!?Jt??%Q@)A??ǘ???1?{????(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?X?? ??!?i\1?P @)?X?? ??1?i\1?P @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-????ƻ?!?&b??5.@)x$(~???1?yb??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ݓ????!y??%4@)?:pΈ??10Pj?(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*A??ǘ???!?{????@)A??ǘ???1?{????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9F??!9@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?ɦE??U? I??????x?&??!???????	!       "	!       *	!       2$	?dv¶.???????T??V-????!}?5^?I??:	!       B	!       J$	?e?????u"%???lxz?,C??!?/?'??R	!       Z$	?e?????u"%???lxz?,C??!?/?'??JCPU_ONLYYF??!9@b Y      Y@q????VU@"?
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?85.356% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 