?($	TD?????_?3?u????????!?V?/?'??$	?p?0@o&???@?j ??@!?L?C])@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???&????d?`T??A7?[ A??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K??????????A?H.?!???Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&mV}??b??jM??St??A?):????Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+?????1w-!??A??????Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????ׁ???S㥛???A*??D???Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??g??s??6?;Nё??A??	h"l??Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??1??%?????z6??A???z6??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??????q????A.?!??u??YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?d?`TR??NbX9???A?"??~j??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?V?/?'??333333??AD????9??YGx$(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
)??0???sh??|???A?46<??Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&? ?rh?????:M??AbX9????Y,e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>?٬?\??F????x??A?Ǻ????Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?/L?
???/L?
F??AS?!?uq??Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<Nё\????c?ZB??A?{??Pk??Y??D????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??z6???	?^)???AJ{?/L???Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~??k??jM??S??AZd;?O???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???QI??=?U????Ao???T???Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?R?!?u??x$(~???AǺ????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ۊ?e??????n????A??z6???Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????-C??6??Am???????Y??ܵ?|??*	fffff?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?~j?t???!?E~?_?@@)??"??~??1؀??j>@:Preprocessing2F
Iterator::ModelO@a????!eO???B@)?C?l????1E#???P8@:Preprocessing2U
Iterator::Model::ParallelMapV2r?鷯??!?<?B?*@)r?鷯??1?<?B?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?^)???!??g?/O@)?c?ZB??1?xS??t$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?????B??!?$Y?9?,@)?~?:pθ?1?t@? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicee?X???!%`????@)e?X???1%`????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??s????!o????2@)?z?G???1.?? ,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?p=
ף??!?S?I?@)?p=
ף??1?S?I?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?I(
?p@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??J??i??߳????-C??6??!333333??	!       "	!       *	!       2$	<?;?o/??j3n	s???m???????!?{??Pk??:	!       B	!       J$	???.????:&I(??w-!?l??!?sF????R	!       Z$	???.????:&I(??w-!?l??!?sF????JCPU_ONLYY?I(
?p@b Y      Y@q?????:@"?
both?Your program is POTENTIALLY input-bound because 51.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?26.9361% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 