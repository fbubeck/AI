?($	;??~y/??X>
qz????????!?Zd;??$	?w???@a??Z?
@???T^\ @!?????1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??z6??????z6??A????H??Yj?q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?A`????r?鷯??A9EGr???Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?;Nё\??A?c?]K??A{?/L?
??Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??镲???ZӼ???A????ׁ??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ͪ?????MbX9??Aꕲq???Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??6???????x???A?Fx$??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????H????V?/???AHP?s??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY??????????A?:pΈ???YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ޓ??Z???&S??:??A?鷯??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?$??C???? ?	??A?3??7???YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
m???{?????MbX??A??&S??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*?????R?!?u??A?"??~j??Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ????S?!?uq??AU???N@??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?g??s?????|гY??A3ı.n???Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;??bX9????A?镲q??Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&?????e?c]??AC??6??Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX????-????A?]K?=??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF??<Nё\???A?	?c??Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??QI????C?i?q???AT㥛? ??Y㥛? ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??/?$????C?l???A?
F%u??Y??\m????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????{??Pk??A^K?=???Y??j+????*	     ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatp_?Q??!?a??B@)?ͪ??V??1?Uv[?A@:Preprocessing2F
Iterator::Model?<,Ԛ???!:??Η??@)P?s???1?Z????2@:Preprocessing2U
Iterator::Model::ParallelMapV2?|?5^???!kCol??)@)?|?5^???1kCol??)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip>yX?5???!? BZQ@)D?l?????1*7?j)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicebX9?ȶ?!兜??c @)bX9?ȶ?1兜??c @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenated]?Fx??!n*?'?,@)e?`TR'??1I???=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapԚ?????!,@?A?1@)??(\?¥?1?G???N@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??????!?o??.?@)??????1?o??.?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9zƌb?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	78?>?????7????????z6??!bX9????	!       "	!       *	!       2$	w$(~????'???`??^K?=???!?:pΈ???:	!       B	!       J$	??ܵ?|??e0G????g??s???!j?q?????R	!       Z$	??ܵ?|??e0G????g??s???!j?q?????JCPU_ONLYYzƌb?@b Y      Y@q.?C ?S@"?
both?Your program is POTENTIALLY input-bound because 51.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?79.0801% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 