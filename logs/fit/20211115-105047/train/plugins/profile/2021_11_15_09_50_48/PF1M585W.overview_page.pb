?($	???ĭ??EI?_G??Ϊ??V???!jM????$	??cS??@1Pޑ?@?????@!r:??('0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$w-!?l??=,Ԛ???A?5^?I??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?q????????B?i??A$???????Y,e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???K7????-?????A	?^)???Y㥛? ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM????&䃞ͪ??A0*??D??Y?/?'??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???h o??`??"????A[Ӽ???Y??|гY??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&mV}??b??\ A?c???A?lV}???Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??m4????i o????Ac?=yX??Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??1??%????ܵ??ATR'?????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M??St$??l	??g???A?ڊ?e???Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??<,Ԛ???H?}8??A?~j?t???Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?J?4?????????Ah"lxz???Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,Ԛ?????*??	??A?D?????Y?V-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?|??O??e?c??A[????<??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"??u???????&S??AX9??v??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+????r?鷯??A??ܵ???Yı.n???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??*:??H??A???????Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_vO????(????A??H.???Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e??a???H?}??A??e??a??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=?U???~??k	???A?? ???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U???N@??~??k	???A?T???N??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ϊ??V????,C????AC?i?q???YHP?s??*	gffff
?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??u????!m?/XB@)~8gDi??1?i??@@:Preprocessing2F
Iterator::Model??ǘ????!?g?Y?A@)???QI???1=?`]Sn5@:Preprocessing2U
Iterator::Model::ParallelMapV2q?-???!Y?C???,@)q?-???1Y?C???,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?{??Pk??!%L?x	P@)I??&??1t?{$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	?^)˰?!e<??G>@)	?^)˰?1e<??G>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?<,Ԛ???!?ʶ?*@)-C??6??1?;?y??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-???????!??+??71@)vOjM??1??GT?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ݓ????!SFTl@)??ݓ????1SFTl@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9(jc??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	 ?lV}???Cڔw???,C????!&䃞ͪ??	!       "	!       *	!       2$	?b??????r鄀??C?i?q???!??ܵ???:	!       B	!       J$	:?o?]??D? ??"??u????!B>?٬???R	!       Z$	:?o?]??D? ??"??u????!B>?٬???JCPU_ONLYY(jc??@b Y      Y@q??y/17?@"?
both?Your program is POTENTIALLY input-bound because 58.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.2156% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 