?($	??aoM???pP?G???Zd;?O???!??y?)??$	???7?@??$?@?.??j&	@!?X???*@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q=
ףp????Pk?w??AӼ????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ǘ????Gx$(??A+??????Y??	h"l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&OjM???l	??g???A??6?[??YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O@a??????@?????An????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??9#J{??f??a????A?ܵ?|???Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?`TR'????ZӼ???A???Mb??Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?i?q????333333??A?j+?????Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??S??^K?=???A??H.?!??Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ڬ?\m???䃞ͪ???A?5?;N???Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??(\??????e?c]??Ash??|???Y2??%䃞?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
;M?O??ꕲq???A?46<??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ݓ????ŏ1w-!??A???~?:??Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?j+?????yX?5?;??A?\m?????Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q??????&S???A@?߾???YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t???%??C???ATR'?????Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):??????|гY??A?J?4??Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??H.???=
ףp=??A??T?????Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)?????z6??Ah??s???Y??y?)??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș?????+??????A)?Ǻ???Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??g??s???V?/?'??Au????Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O???bX9?ȶ?A?G?z???YT㥛? ??*	33333s?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatk?w??#??!?wÏ??@@)o?ŏ1??1{??i%?>@:Preprocessing2F
Iterator::Model??~j?t??!???-s?B@)l	??g???1[?~?uh5@:Preprocessing2U
Iterator::Model::ParallelMapV2K?46??!??$?p40@)K?46??1??$?p40@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??g??s??!`=.Ҍ1O@)????????1ҌQ???$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?? ?rh??!Y?&$?+@)㥛? ???14????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceT㥛? ??!	?=???@)T㥛? ??1	?=???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA??ǘ???!?Ĭ?:2@)??_?L??1?`Z??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*ŏ1w-!??! ?!?}?@)ŏ1w-!??1 ?!?}?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?:?_߈@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?:?;?????ה??bX9?ȶ?!?V?/?'??	!       "	!       *	!       2$	B??^YR???k??B???G?z???!h??s???:	!       B	!       J$	\μp|???փf%܂??D???J??!S?!?uq??R	!       Z$	\μp|???փf%܂??D???J??!S?!?uq??JCPU_ONLYY?:?_߈@b Y      Y@q_??dT@"?
both?Your program is POTENTIALLY input-bound because 49.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?81.5735% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 