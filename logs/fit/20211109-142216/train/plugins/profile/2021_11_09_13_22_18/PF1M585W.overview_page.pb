?)$	+??"4??,?#???QI??&??!???h o??$	????q@D??9?@?J*???@!?+?1[2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??? ?r?????????A*??D???Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?`TR'???3??7???AjM????Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t???	?^)???A}?5^?I??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1?????????A??T?????Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U??A?c?]K??A??e?c]??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?Q????t?V??A?	h"lx??Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0??Tt$?????A??o_??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)?????w??#???A??T?????YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx??z?):????AZ??ڊ???YDio??ɤ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??C?l??1?*????AjM????Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?=yX?5??}гY????A?Pk?w???YHP?sע?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???h o??????(??A?G?z???Y(??y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&3ı.n????R?!?u??A}?5^?I??YP?s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?U??????????9#??Aı.n???Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??q????`vOj??AX9??v???Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???{????:??H???A??JY?8??Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\?C?????-????A?s????Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ???5?8EGr??A?O??n??Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&A??ǘ???/?$???ARI??&???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ??2w-!???A??ZӼ???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?QI??&??a??+e??A?E??????Y??ǘ????*	33333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM?J???!?fy??A@);?O??n??1???"@@:Preprocessing2F
Iterator::Model?!??u???!???z?FB@)?b?=y??1P???l5@:Preprocessing2U
Iterator::Model::ParallelMapV2H?z?G??!2???gA.@)H?z?G??12???gA.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??_vO??!
Rs?3?O@)
ףp=
??1"?Y??+$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?5?;Nѱ?!???_2@)?5?;Nѱ?1???_2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?O??e??!?????,@)?ʡE????1??AU?8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?L?J???!+?ƝgO2@)P?s???19????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*a??+e??!\ JVp;@)a??+e??1\ JVp;@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t51.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	+??7???????O?[?????????!1?*????	!       "	!       *	!       2$	??_vO??QmKw7????E??????!}?5^?I??:	!       B	!       J$	V??¤?<??????ǘ????!(??y??R	!       Z$	V??¤?<??????ǘ????!(??y??JCPU_ONLYY????@b Y      Y@q??wEQ=@"?	
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t51.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?29.3175% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 